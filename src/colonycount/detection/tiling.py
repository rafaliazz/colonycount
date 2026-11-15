"""Overlapping tiles, seam ownership and NMS merge.

Each box is kept only in the tile whose core (tile minus half the overlap on inner sides) holds
its centre, so a colony on a seam is counted once; NMS then merges what remains across tiles.
"""

import math

import cv2
import numpy as np

from colonycount._env import env_float
from colonycount.detection.runner import infer


def _tiles(n: int, tile: int, overlap: int) -> tuple[list[int], list[float]]:
    """Tile starts along one axis, and core bounds: tile i owns centres in [b[i], b[i+1])."""
    if n <= tile:
        return [0], [-math.inf, math.inf]
    starts = [*range(0, n - tile, tile - overlap), n - tile]
    mids = [
        (a + b + tile) / 2 for a, b in zip(starts[:-1], starts[1:], strict=True)
    ]  # middle of each overlap
    return starts, [-math.inf, *mids, math.inf]


def detect_tiled(session, crop: np.ndarray, tile: int, overlap: int):
    """Boxes (x1, y1, x2, y2 in crop pixels) and scores, each owned by exactly one tile."""
    h, w = crop.shape[:2]
    ys, ybounds = _tiles(h, tile, overlap)
    xs, xbounds = _tiles(w, tile, overlap)
    all_boxes, all_scores = [np.zeros((0, 4), np.float32)], [np.zeros(0, np.float32)]
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            boxes, scores = infer(session, crop[y : y + tile, x : x + tile], tile)
            boxes = boxes + [x, y, x, y]
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            own = (
                (xbounds[j] <= cx)
                & (cx < xbounds[j + 1])
                & (ybounds[i] <= cy)
                & (cy < ybounds[i + 1])
            )
            all_boxes.append(boxes[own])
            all_scores.append(scores[own])
    return np.concatenate(all_boxes).astype(np.float64), np.concatenate(all_scores)


def nms(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Sorted indices of the boxes NMS keeps (scores are already thresholded)."""
    iou = env_float("COLONYCOUNT_NMS_IOU", 0.4, 0, 1)
    xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]])
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou)
    return np.sort(np.asarray(idx, dtype=int).reshape(-1))
