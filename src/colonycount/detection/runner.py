"""ONNX Runtime session loading, model-contract check and per-tile inference.

Model contract (every built-in model's ONNX file must follow it; task 2.5 exports to it),
checked when the session loads:

- input ``images``: float32 ``[1, 3, T, T]``, RGB, values 0..1. ``T`` is fixed in the model and
  is the tile size used at inference. Edge tiles are zero-padded bottom/right.
- outputs ``boxes``: float32 ``[N, 4]`` as x1, y1, x2, y2 in tile pixels, and
  ``scores``: float32 ``[N]`` in [0, 1]. ``N`` may be 0. A per-pass cap on ``N`` is fine:
  tiling keeps the per-image count uncapped.
"""

from functools import lru_cache

import numpy as np
import onnxruntime as ort


@lru_cache(maxsize=4)
def load_session(path: str) -> ort.InferenceSession:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if [i.name for i in inputs] != ["images"]:
        raise ValueError(f"{path}: expected one input 'images', got {[i.name for i in inputs]}")
    if inputs[0].type != "tensor(float)":
        raise ValueError(f"{path}: input 'images' must be float32, got {inputs[0].type}")
    shape = inputs[0].shape  # dynamic dims come back as str or None
    if not (
        len(shape) == 4 and shape[1] == 3 and isinstance(shape[2], int) and shape[2] == shape[3]
    ):
        raise ValueError(f"{path}: input 'images' must be [1, 3, T, T] with a fixed T, got {shape}")
    missing = {"boxes", "scores"} - {o.name for o in session.get_outputs()}
    if missing:
        raise ValueError(f"{path}: missing output(s) {sorted(missing)}")
    return session


def tile_size(session) -> int:
    """The model's tile size T, from its ``images`` input shape."""
    return session.get_inputs()[0].shape[2]


def infer(session, part: np.ndarray, tile: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on one uint8 RGB tile (at most tile x tile); boxes in tile pixels."""
    patch = np.zeros((tile, tile, 3), np.float32)
    patch[: part.shape[0], : part.shape[1]] = part / 255.0
    boxes, scores = session.run(["boxes", "scores"], {"images": patch.transpose(2, 0, 1)[None]})
    return boxes.reshape(-1, 4), scores.reshape(-1)
