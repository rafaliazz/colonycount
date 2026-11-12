"""Colony detection: plate localisation, tiled ONNX inference, seam-aware merge.

Model contract (every built-in model's ONNX file must follow it; task 2.5 exports to it):

- input ``images``: float32 ``[1, 3, T, T]``, RGB, values 0..1, ``T`` = tile size
  (``TILE_SIZE`` unless ``tile_size=`` is passed). Edge tiles are zero-padded bottom/right.
- outputs ``boxes``: float32 ``[N, 4]`` as x1, y1, x2, y2 in tile pixels, and
  ``scores``: float32 ``[N]`` in [0, 1]. ``N`` may be 0. A per-pass cap on ``N`` is fine:
  tiling keeps the per-image count uncapped.

Pipeline: find the plate (Hough circle on a downscaled copy; fallback: whole image,
``plate_found=False``), crop to it, run the model on overlapping tiles at native resolution,
keep each box only in the tile whose core (tile minus half the overlap on inner sides) holds its
centre, apply the threshold, drop centres outside the plate, then NMS across all tiles.
"""

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from colonycount import weights

# ponytail: tile size, overlap and threshold are placeholders until tasks 2.3/2.5 fix them.
TILE_SIZE = 640
TILE_OVERLAP = 64  # colonies wider than this can be split across a seam
DEFAULT_THRESHOLD = 0.5
NMS_IOU = 0.5
PLATE_SEARCH_SIZE = 512  # longest side of the downscaled copy used to find the plate


@dataclass(frozen=True)
class Detection:
    x: float  # centre, image pixels
    y: float
    diameter: float  # mean of box width and height, pixels
    confidence: float


@dataclass(frozen=True)
class Result:
    detections: list[Detection]
    plate_found: bool
    threshold: float
    model: str
    model_version: str

    @property
    def total(self) -> int:
        return len(self.detections)


def count(
    image: str | os.PathLike | np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = "public",
    model_path: str | os.PathLike | None = None,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> Result:
    """Count colonies in a plate photo (path or HxW / HxWx3 RGB array)."""
    img = load_image(image)
    path, version = weights.resolve(model, model_path)
    session = _session(str(path))

    plate = find_plate(img)
    h, w = img.shape[:2]
    if plate:
        cx, cy, r = plate
        x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
        x1, y1 = min(w, math.ceil(cx + r) + 1), min(h, math.ceil(cy + r) + 1)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    boxes, scores = _detect_tiled(session, img[y0:y1, x0:x1], tile_size, overlap)
    boxes += [x0, y0, x0, y0]

    keep = scores >= threshold
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    if plate:
        keep &= np.hypot(centres[:, 0] - cx, centres[:, 1] - cy) <= r
    boxes, scores, centres = boxes[keep], scores[keep], centres[keep]
    if len(boxes):
        xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]])
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, NMS_IOU)  # already thresholded
        idx = np.sort(np.asarray(idx, dtype=int).reshape(-1))
        boxes, scores, centres = boxes[idx], scores[idx], centres[idx]
    sizes = (boxes[:, 2:] - boxes[:, :2]).mean(axis=1)
    detections = [
        Detection(float(x), float(y), float(d), float(s))
        for (x, y), d, s in zip(centres, sizes, scores, strict=True)
    ]
    return Result(detections, plate is not None, threshold, model, version)


def load_image(image: str | os.PathLike | np.ndarray) -> np.ndarray:
    """Return an HxWx3 uint8 RGB array from a path (JPEG/PNG/TIFF) or an array."""
    if isinstance(image, np.ndarray):
        arr = image
    else:
        data = np.fromfile(Path(image), np.uint8)  # imread can't open non-ASCII paths on Windows
        arr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if arr is None:
            raise ValueError(f"not a readable image: {image}")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        arr = arr[..., :3] if arr.shape[2] != 1 else arr[..., 0]
    if arr.ndim not in (2, 3):
        raise ValueError(f"expected an HxW or HxWx3 image array, got shape {arr.shape}")
    if arr.dtype == np.uint16:
        arr = (arr // 257).astype(np.uint8)
    elif arr.dtype.kind == "f":
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        raise ValueError(f"unsupported image dtype {arr.dtype}")
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB) if arr.ndim == 2 else np.ascontiguousarray(arr)


def find_plate(img: np.ndarray) -> tuple[float, float, float] | None:
    """Return the plate circle (cx, cy, r) in image pixels, or None if no rim is found."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    scale = min(1.0, PLATE_SEARCH_SIZE / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 9)  # wipes out colonies, keeps the rim
    side = min(small.shape)
    # ponytail: tuned on synthetic plates only, first (strongest) circle wins and a plate's
    # inner vs outer rim isn't told apart; retune on the public test photos (task 2.3).
    circles = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=side,
        param1=100,
        param2=50,
        minRadius=int(0.25 * side),
        maxRadius=int(0.75 * max(small.shape)),
    )
    if circles is None:
        return None
    cx, cy, r = circles[0, 0] / scale
    return float(cx), float(cy), float(r)


@lru_cache(maxsize=4)
def _session(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _tiles(n: int, tile: int, overlap: int) -> tuple[list[int], list[float]]:
    """Tile starts along one axis, and core bounds: tile i owns centres in [b[i], b[i+1])."""
    if n <= tile:
        return [0], [-math.inf, math.inf]
    starts = [*range(0, n - tile, tile - overlap), n - tile]
    mids = [
        (a + b + tile) / 2 for a, b in zip(starts[:-1], starts[1:], strict=True)
    ]  # middle of each overlap
    return starts, [-math.inf, *mids, math.inf]


def _detect_tiled(session, crop: np.ndarray, tile: int, overlap: int):
    h, w = crop.shape[:2]
    ys, ybounds = _tiles(h, tile, overlap)
    xs, xbounds = _tiles(w, tile, overlap)
    all_boxes, all_scores = [np.zeros((0, 4), np.float32)], [np.zeros(0, np.float32)]
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = np.zeros((tile, tile, 3), np.float32)
            part = crop[y : y + tile, x : x + tile]
            patch[: part.shape[0], : part.shape[1]] = part / 255.0
            boxes, scores = session.run(
                ["boxes", "scores"], {"images": patch.transpose(2, 0, 1)[None]}
            )
            boxes = boxes.reshape(-1, 4) + [x, y, x, y]
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            own = (
                (xbounds[j] <= cx)
                & (cx < xbounds[j + 1])
                & (ybounds[i] <= cy)
                & (cy < ybounds[i + 1])
            )
            all_boxes.append(boxes[own])
            all_scores.append(scores.reshape(-1)[own])
    return np.concatenate(all_boxes).astype(np.float64), np.concatenate(all_scores)
