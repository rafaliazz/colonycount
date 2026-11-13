"""YOLOv8 baseline: COCO -> YOLO conversion and config (no torch/ultralytics needed), plus ONNX
parity vs Ultralytics' predict(), only where ultralytics, a trained run and the val tiles exist."""

import json
import os
from pathlib import Path

import pytest
import train_yolo

from colonycount._env import env_path


def test_coco_to_yolo_writes_normalised_labels_and_data_yaml(tmp_path):
    tiles = tmp_path / "tiles_640"
    for split in train_yolo.SPLITS:
        (tiles / split).mkdir(parents=True)
        (tiles / split / "a.jpg").write_bytes(b"jpg")
        (tiles / split / "empty.jpg").write_bytes(b"jpg")
        coco = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 640, "height": 320},
                {"id": 2, "file_name": "empty.jpg", "width": 640, "height": 640},
            ],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [64, 32, 128, 64]}],
            "categories": [{"id": 1, "name": "colony"}],
        }
        (tiles / split / "_annotations.coco.json").write_text(json.dumps(coco))
    out = tmp_path / "yolo_640"
    data_yaml = train_yolo.coco_to_yolo(tiles, out)
    train_yolo.coco_to_yolo(tiles, out)  # idempotent
    label = (out / "train" / "labels" / "a.txt").read_text()
    assert label == "0 0.200000 0.200000 0.200000 0.200000\n"  # cx, cy, w, h / tile w, h
    assert (out / "valid" / "labels" / "empty.txt").read_text() == ""
    assert (out / "valid" / "images" / "a.jpg").read_bytes() == b"jpg"
    text = data_yaml.read_text()
    assert "train: train/images" in text and "val: valid/images" in text and "0: colony" in text


def test_config_new_run_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYCOUNT_SEED", "7")
    monkeypatch.delenv("COLONYCOUNT_TRAIN_BATCH_SIZE", raising=False)
    monkeypatch.delenv("COLONYCOUNT_TRAIN_GRAD_ACCUM", raising=False)
    args = train_yolo.parse_args(["--size", "m", "--data-dir", str(tmp_path), "--epochs", "3"])
    c = train_yolo.build_config(args, tmp_path / "data.yaml")
    assert (c["epochs"], c["imgsz"], c["seed"], c["batch"], c["single_cls"]) == (3, 640, 7, -1, 1)
    assert c["max_det"] >= 300 and c["project"] == str(tmp_path / "runs" / "yolov8m")
    assert c["nbs"] == 16  # RF-DETR's effective batch: 2 x grad accum 8
    monkeypatch.setenv("COLONYCOUNT_TRAIN_BATCH_SIZE", "4")
    monkeypatch.setenv("COLONYCOUNT_TRAIN_GRAD_ACCUM", "3")
    assert train_yolo.build_config(args, tmp_path / "data.yaml")["nbs"] == 12
    assert train_yolo.build_config(
        train_yolo.parse_args(["--size", "n", "--batch", "0.7"]), tmp_path
    )["batch"] == pytest.approx(0.7)

    last = tmp_path / "last.pt"
    last.write_bytes(b"")
    args = train_yolo.parse_args(["--size", "n", "--resume", str(last), "--batch", "8"])
    assert train_yolo.build_config(args, tmp_path) == {
        "resume": str(last),
        "batch": 8,
        "workers": 4,
    }
    with pytest.raises(SystemExit):
        train_yolo.parse_args(["--size", "n", "--resume", str(tmp_path / "nope.pt")])
    with pytest.raises(SystemExit):
        train_yolo.parse_args(["--size", "x"])


def test_comparable_config_and_no_accumulation_warning(capsys):
    from types import SimpleNamespace as NS

    a = NS(epochs=12, imgsz=640, seed=0, patience=5, nbs=16, optimizer="auto")
    t = NS(args=a, batch_size=3, optimizer=NS(), amp=True)  # no .accumulate: computed
    c = train_yolo.comparable(t)
    assert (c["nbs"], c["batch"], c["accumulate"], c["amp"]) == (16, 3, 5, True)
    assert "WARNING" not in capsys.readouterr().out
    train_yolo.comparable(NS(args=a, batch_size=32, optimizer=None, amp=False, accumulate=1))
    assert "WARNING" in capsys.readouterr().out


data = env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
runs = sorted((data / "runs" / "yolov8n").glob("*/weights/best.pt"))
weights = Path(os.environ.get("COLONYCOUNT_YOLO_WEIGHTS", runs[-1] if runs else "none"))
tiles = data / "public/tiles_640/valid"


@pytest.mark.skipif(not weights.is_file() or not tiles.is_dir(), reason="no YOLO run or val tiles")
def test_yolo_onnx_matches_ultralytics(tmp_path):
    pytest.importorskip("ultralytics")
    import export_yolo_onnx
    import onnxruntime as ort

    out = tmp_path / "m.onnx"
    yolo, tile = export_yolo_onnx.export(weights, out)
    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert [(i.name, i.shape) for i in s.get_inputs()] == [("images", [1, 3, tile, tile])]
    assert [o.name for o in s.get_outputs()] == ["boxes", "scores"]
    p = export_yolo_onnx.parity(yolo, tile, out, export_yolo_onnx.val_tiles(tiles, 4, tile))
    assert p["ok"] and p["n_boxes"] > 0, p
