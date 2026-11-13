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
