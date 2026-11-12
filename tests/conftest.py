"""Placeholder ONNX detector and synthetic plate images (no dataset files in the repo)."""

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

MAX_PER_PASS = 300  # like a real detector's query/anchor cap
BOX = 4.0  # half side of every box the placeholder emits


def build_placeholder_model(path):
    """Bright-spot detector following the detection/runner.py contract.

    A pixel is a colony centre if it is the max of its 7x7 neighbourhood and brighter than 0.5
    (channel mean); its score is that brightness. At most MAX_PER_PASS boxes per pass (top-k).
    """
    c = helper.make_tensor
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["gray"], axes=[1], keepdims=1),
        helper.make_node("MaxPool", ["gray"], ["pooled"], kernel_shape=[7, 7], pads=[3] * 4),
        helper.make_node("Equal", ["gray", "pooled"], ["is_max"]),
        helper.make_node("Greater", ["gray", "half"], ["bright"]),
        helper.make_node("And", ["is_max", "bright"], ["peak"]),
        helper.make_node("NonZero", ["peak"], ["nz"]),  # [4, N]: n, c, y, x
        helper.make_node("Transpose", ["nz"], ["idx"], perm=[1, 0]),
        helper.make_node("GatherND", ["gray", "idx"], ["all_scores"]),
        helper.make_node("Shape", ["all_scores"], ["n"]),
        helper.make_node("Min", ["n", "cap"], ["k"]),
        helper.make_node("TopK", ["all_scores", "k"], ["scores", "top"]),
        helper.make_node("Gather", ["nz", "two"], ["ys_all"], axis=0),
        helper.make_node("Gather", ["nz", "three"], ["xs_all"], axis=0),
        helper.make_node("Gather", ["ys_all", "top"], ["ys_i"]),
        helper.make_node("Gather", ["xs_all", "top"], ["xs_i"]),
        helper.make_node("Cast", ["ys_i"], ["ys"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["xs_i"], ["xs"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["xs", "one"], ["x"]),
        helper.make_node("Unsqueeze", ["ys", "one"], ["y"]),
        helper.make_node("Sub", ["x", "r"], ["x1"]),
        helper.make_node("Sub", ["y", "r"], ["y1"]),
        helper.make_node("Add", ["x", "r"], ["x2"]),
        helper.make_node("Add", ["y", "r"], ["y2"]),
        helper.make_node("Concat", ["x1", "y1", "x2", "y2"], ["boxes"], axis=1),
    ]
    consts = [
        c("half", TensorProto.FLOAT, [], [0.5]),
        c("cap", TensorProto.INT64, [1], [MAX_PER_PASS]),
        c("two", TensorProto.INT64, [], [2]),
        c("three", TensorProto.INT64, [], [3]),
        c("one", TensorProto.INT64, [1], [1]),
        c("r", TensorProto.FLOAT, [], [BOX]),
    ]
    graph = helper.make_graph(
        nodes,
        "placeholder",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, "H", "W"])],
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, ["N", 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["N"]),
        ],
        initializer=consts,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def make_plate(
    n=40, *, size=512, radius=200, plate=True, labels=False, spacing=20, peaks=(150, 255), seed=0
):
    """Synthetic plate photo: agar disc on a dark bench, Gaussian-dot colonies, optional labels.

    Returns (RGB uint8 image, list of (x, y) colony centres).
    """
    rng = np.random.default_rng(seed)
    agar = 100
    img = np.full((size, size), agar if not plate else 40, np.uint8)
    c = size // 2
    if plate:
        cv2.circle(img, (c, c), radius, agar, -1)
        inner = radius - 12
    if labels:  # bright handwriting-like text on the bench, in the plate's bounding-box corners
        for org in [(4, 20), (c - radius + 2, c - radius + 30), (c + radius - 60, c + radius - 6)]:
            cv2.putText(img, "1e-4", org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 250, 1)
    img = img.astype(np.float64)
    g = np.arange(12, size - 12, spacing)
    pts = [(x, y) for y in g for x in g if not plate or (x - c) ** 2 + (y - c) ** 2 <= inner**2]
    assert len(pts) >= n, f"only {len(pts)} slots for {n} colonies"
    pts = [pts[i] for i in rng.choice(len(pts), n, replace=False)]
    yy, xx = np.mgrid[-6:7, -6:7]
    blob = np.exp(-(xx**2 + yy**2) / 8.0)  # sigma 2 px: unique peak, ~8 px wide
    for x, y in pts:
        peak = rng.uniform(*peaks)
        win = img[y - 6 : y + 7, x - 6 : x + 7]
        np.maximum(win, agar + (peak - agar) * blob, out=win)
    img = np.clip(img.round(), 0, 255).astype(np.uint8)
    return np.dstack([img] * 3), pts


@pytest.fixture(scope="session")
def model_file(tmp_path_factory):
    return build_placeholder_model(tmp_path_factory.mktemp("model") / "placeholder.onnx")


@pytest.fixture
def local_model(model_file, monkeypatch):
    monkeypatch.setenv("COLONYCOUNT_MODEL_PATH", str(model_file))
    return model_file
