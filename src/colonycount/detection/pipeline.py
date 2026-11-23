"""count(): image in, colonies out.

Pipeline: find the plate (Hough circle on a downscaled copy; fallback: whole image,
``plate_found=False``), crop to it, run the model on overlapping tiles at native resolution,
keep each box only in the tile whose core holds its centre, apply the threshold, drop centres
outside the plate, then NMS across all tiles.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from colonycount._env import env_float, env_int, env_str
from colonycount.detection.plate import find_plate
from colonycount.detection.runner import load_session, tile_size
from colonycount.detection.tiling import detect_tiled, nms
from colonycount.models import registry


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
    threshold: float | None = None,
    model: str | None = None,
    model_path: str | os.PathLike | None = None,
) -> Result:
    """Count colonies in a plate photo (path or HxW / HxWx3 RGB array).

    ``model`` defaults to ``COLONYCOUNT_MODEL``; ``threshold`` to ``COLONYCOUNT_THRESHOLD``, else
    the model's own default (0.5 for a local ONNX file);
    the tile size comes from the model, the tile overlap from ``COLONYCOUNT_TILE_OVERLAP``.
    """
    model = model or env_str("COLONYCOUNT_MODEL", "rfdetr-nano")
    if threshold is None:
        threshold = registry.threshold(model, model_path)
    img = load_image(image)
    path, version = registry.resolve(model, model_path)
    session = load_session(str(path))
    tile = tile_size(session)
    # ponytail: 64 is a placeholder until task 2.3; colonies wider than it can split on a seam.
    overlap = env_int("COLONYCOUNT_TILE_OVERLAP", 64, lo=0)
    if overlap >= tile:
        raise ValueError(f"COLONYCOUNT_TILE_OVERLAP ({overlap}) must be below the tile size {tile}")

    plate = find_plate(img)
    h, w = img.shape[:2]
    if plate:
        cx, cy, r = plate
        x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
        x1, y1 = min(w, math.ceil(cx + r) + 1), min(h, math.ceil(cy + r) + 1)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    crop = img[y0:y1, x0:x1]
    upscale = env_float("COLONYCOUNT_UPSCALE", 1.0, lo=1e-6)
    if upscale != 1.0:  # test-time upscaling: small colonies look bigger to the model
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)
    boxes, scores = detect_tiled(session, crop, tile, overlap)
    if upscale != 1.0:
        boxes /= upscale
    boxes += [x0, y0, x0, y0]

    keep = scores >= threshold
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    if plate:
        keep &= np.hypot(centres[:, 0] - cx, centres[:, 1] - cy) <= r
    boxes, scores, centres = boxes[keep], scores[keep], centres[keep]
    if len(boxes):
        idx = nms(boxes, scores)
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
