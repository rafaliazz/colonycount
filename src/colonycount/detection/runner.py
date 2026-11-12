"""ONNX Runtime session loading and per-tile inference.

Model contract (every built-in model's ONNX file must follow it; task 2.5 exports to it):

- input ``images``: float32 ``[1, 3, T, T]``, RGB, values 0..1, ``T`` = tile size
  (``tiling.TILE_SIZE`` unless ``tile_size=`` is passed). Edge tiles are zero-padded
  bottom/right.
- outputs ``boxes``: float32 ``[N, 4]`` as x1, y1, x2, y2 in tile pixels, and
  ``scores``: float32 ``[N]`` in [0, 1]. ``N`` may be 0. A per-pass cap on ``N`` is fine:
  tiling keeps the per-image count uncapped.
"""

from functools import lru_cache

import numpy as np
import onnxruntime as ort


@lru_cache(maxsize=4)
def load_session(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def infer(session, part: np.ndarray, tile: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on one uint8 RGB tile (at most tile x tile); boxes in tile pixels."""
    patch = np.zeros((tile, tile, 3), np.float32)
    patch[: part.shape[0], : part.shape[1]] = part / 255.0
    boxes, scores = session.run(["boxes", "scores"], {"images": patch.transpose(2, 0, 1)[None]})
    return boxes.reshape(-1, 4), scores.reshape(-1)
