"""colony-detection spec scenarios (model registry and offline ones are in test_registry.py)."""

import conftest
import cv2
import numpy as np
import pytest
from conftest import make_plate

import colonycount
from colonycount.detection.plate import find_plate
from colonycount.detection.runner import load_session, tile_size
from colonycount.detection.tiling import detect_tiled


def write(path, img):
    assert cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def matched(result, truth, tol=1.5):
    got = np.array([(d.x, d.y) for d in result.detections])
    return all(np.hypot(*(got - p).T).min() <= tol for p in truth)


@pytest.mark.parametrize("fmt", ["png", "jpg", "tif", "gray-png", "rgb-array", "gray-array"])
def test_typical_plate(fmt, tmp_path, local_model):
    img, truth = make_plate()
    if fmt == "rgb-array":
        image = img
    elif fmt == "gray-array":
        image = img[..., 0]
    elif fmt == "gray-png":
        image = tmp_path / "p.png"
        assert cv2.imwrite(str(image), img[..., 0])
    else:
        image = write(tmp_path / f"p.{fmt}", img)
    r = colonycount.count(image)
    assert r.plate_found
    assert r.total == len(r.detections) == len(truth)
    assert all(r.threshold <= d.confidence <= 1 for d in r.detections)
    assert all(6 <= d.diameter <= 10 for d in r.detections)
    assert matched(r, truth)
    assert (r.model, r.model_version) == ("rfdetr-nano", "local")


def test_unsupported_input(tmp_path, local_model):
    bad = tmp_path / "notes.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="notes.jpg"):
        colonycount.count(bad)
    with pytest.raises(FileNotFoundError, match="missing.png"):
        colonycount.count(tmp_path / "missing.png")


def test_label_outside_plate(local_model):
    img, truth = make_plate(labels=True)
    raw, _ = detect_tiled(load_session(str(local_model)), img, 640, 64)
    centres = (raw[:, :2] + raw[:, 2:]) / 2
    assert (np.hypot(*(centres - 256).T) > 200).any()  # the model does fire on the labels
    r = colonycount.count(img)
    assert r.plate_found and r.total == len(truth) and matched(r, truth)
    assert all(np.hypot(d.x - 256, d.y - 256) <= 205 for d in r.detections)


def test_close_up_without_rim(local_model):
    img, truth = make_plate(plate=False, size=400)
    r = colonycount.count(img)
    assert not r.plate_found
    assert r.total == len(truth) and matched(r, truth)


def test_raising_threshold_never_increases_count(local_model):
    img, _ = make_plate(n=80, peaks=(130, 255), seed=3)
    counts = [colonycount.count(img, threshold=t).total for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_crowded_plate_not_capped(placeholder, monkeypatch):
    img, truth = make_plate(n=800, size=560, radius=250, spacing=14)
    one_pass = colonycount.count(img, model_path=placeholder(1024))
    assert one_pass.total == conftest.MAX_PER_PASS  # one pass is capped
    monkeypatch.setenv("COLONYCOUNT_TILE_OVERLAP", "32")
    r = colonycount.count(img, model_path=placeholder(160))
    assert r.total == 800 and matched(r, truth)


def test_tile_seams_count_once(placeholder, monkeypatch):
    img, truth = make_plate(n=60, spacing=16, seed=1)
    monkeypatch.setenv("COLONYCOUNT_TILE_OVERLAP", "24")
    r = colonycount.count(img, model_path=placeholder(96))
    assert r.total == len(truth) and matched(r, truth)


@pytest.mark.parametrize("tile", [96, 160])
def test_tile_size_comes_from_the_model(placeholder, tile):
    assert tile_size(load_session(str(placeholder(tile)))) == tile


def test_overlap_must_be_below_tile_size(placeholder, monkeypatch):
    monkeypatch.setenv("COLONYCOUNT_TILE_OVERLAP", "96")
    with pytest.raises(ValueError, match="COLONYCOUNT_TILE_OVERLAP"):
        colonycount.count(make_plate()[0], model_path=placeholder(96))


def test_find_plate_red_disc_padded():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.circle(img, (260, 250), 190, (255, 30, 30), -1)  # red-lit plate: weak luma edge
    cx, cy, r = find_plate(img)
    assert abs(cx - 260) < 5 and abs(cy - 250) < 5 and abs(r - 190 * 1.05) < 8


def test_upscale_maps_boxes_back(monkeypatch):
    from colonycount.detection import pipeline

    seen = []

    def stub(session, crop, tile, overlap):
        seen.append(crop.shape[:2])
        return np.array([[20.0, 20.0, 40.0, 40.0]]), np.array([0.9])

    monkeypatch.setattr(pipeline, "detect_tiled", stub)
    monkeypatch.setattr(pipeline, "find_plate", lambda img: None)
    monkeypatch.setattr(pipeline, "load_session", lambda path: None)
    monkeypatch.setattr(pipeline, "tile_size", lambda session: 640)
    monkeypatch.setenv("COLONYCOUNT_UPSCALE", "2")
    r = colonycount.count(np.zeros((100, 120, 3), np.uint8), model_path=__file__)
    assert seen == [(200, 240)]
    assert (r.detections[0].x, r.detections[0].y, r.detections[0].diameter) == (15.0, 15.0, 10.0)
