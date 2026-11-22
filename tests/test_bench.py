"""Metric functions of bench/evaluate.py on tiny synthetic inputs; no data or model needed."""

import evaluate
import numpy as np
import pytest


def test_mdape_skips_empty_plates_and_takes_the_median():
    assert evaluate.mdape([110, 90, 50, 3], [100, 100, 100, 0]) == pytest.approx(10.0)
    assert evaluate.mdape([5], [0]) is None


def test_match_is_one_to_one_within_radius():
    gt = np.array([[0.0, 0.0], [10.0, 0.0]])
    r = np.array([2.0, 2.0])
    # two predictions near the first colony count once; one is too far from anything
    pred = np.array([[0.5, 0.0], [1.0, 0.0], [10.0, 3.0]])
    assert evaluate.match(pred, gt, r) == 1
    # nearest pair wins: the closer prediction takes colony 0, the other still takes colony 1
    assert evaluate.match(np.array([[1.5, 0.0], [9.0, 0.0]]), gt, r) == 2
    assert evaluate.match(np.zeros((0, 2)), gt, r) == 0


def test_summarize_thresholds_and_buckets():
    plate = {
        "file": "a.jpg",
        "source": "smartphone",
        "gt": 2,
        "gt_xy": np.array([[0.0, 0.0], [10.0, 0.0]]),
        "gt_r": np.array([2.0, 2.0]),
        "gt_outside": 0,
        "plate_found": True,
        "conf": [0.9, 0.6, 0.3],
        "xy": [(0.0, 0.0), (10.0, 0.0), (50.0, 50.0)],
    }
    lo, hi = evaluate.summarize([plate], 0.2), evaluate.summarize([plate], 0.5)
    assert (lo["bias"], lo["precision"], lo["recall"]) == (1, pytest.approx(2 / 3, abs=1e-4), 1)
    assert (hi["mae"], hi["precision"], hi["recall"]) == (0, 1, 1)
    assert hi["n_30_300"] == 0 and hi["mdape_30_300"] is None and hi["mdape_all"] == 0


def test_rescore_matches_evaluate_on_a_synthetic_cache(tmp_path):
    import rescore

    gt = np.array([[-2.0, -2.0, 4.0, 4.0], [8.0, -2.0, 4.0, 4.0]])  # xywh, centres (0,0), (10,0)
    # two overlapping boxes on colony 0 (NMS keeps the better), one on colony 1, one low-score,
    # one far outside the plate circle
    boxes = np.array(
        [[-2, -2, 2, 2], [-2.2, -2, 2, 2], [8, -2, 12, 2], [30, 30, 34, 34], [100, 100, 104, 104]]
    )
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.9])
    plate = {"file": "a.jpg", "source": "s", "gt_boxes": gt, "raw": (boxes, scores)}
    plate |= {"found": (5.0, 0.0, 60.0), "size": (200, 200)}
    (tmp_path / "m.onnx").write_bytes(b"x")
    evaluate.save_detections(tmp_path / "c.npz", [plate], [1.0], tmp_path / "m.onnx", "val", False)
    head, raw = rescore.load(tmp_path / "c.npz")
    got = evaluate.report(rescore.rescore(raw, 0.2, True), "val", False, head["model"])
    # what evaluate.run builds after count(): NMS-merged, plate-restricted detections >= 0.2
    live = {k: plate[k] for k in ("file", "source")} | {"gt": 2, "gt_outside": 0}
    live |= {"gt_xy": gt[:, :2] + gt[:, 2:] / 2, "gt_r": np.array([2.0, 2.0])}
    live |= {"plate_found": True, "conf": [0.9, 0.7, 0.3]}
    live["xy"] = [(0.0, 0.0), (10.0, 0.0), (32.0, 32.0)]
    assert got == evaluate.report([live], "val", False, "m.onnx")


def test_report_at_a_fixed_threshold_instead_of_the_sweep():
    plate = {
        "file": "a.jpg",
        "source": "s",
        "gt": 100,
        "gt_xy": np.zeros((0, 2)),
        "gt_r": np.zeros(0),
        "gt_outside": 0,
        "plate_found": True,
        "conf": [0.9] * 90 + [0.3] * 10 + [0.25] * 10,
        "xy": [(0.0, 0.0)] * 110,
    }
    swept = evaluate.report([plate], "val", False, "m")
    fixed = evaluate.report([plate], "test", False, "m", fixed=0.42)
    assert (swept["threshold_source"], fixed["threshold_source"]) == ("val-sweep", "fixed")
    assert fixed["threshold"] == 0.42 and fixed["by_source"]["overall"]["bias"] == -10
    assert fixed["threshold_curve"].keys() == swept["threshold_curve"].keys() | {"0.42"}


def test_layout_default_is_public_and_dataset_dir_overrides(tmp_path):
    assert evaluate.layout(tmp_path, None) == (tmp_path / "public" / "annotations", tmp_path)
    assert evaluate.layout(tmp_path, tmp_path / "agar") == (
        tmp_path / "agar" / "annotations",
        tmp_path / "agar",
    )
