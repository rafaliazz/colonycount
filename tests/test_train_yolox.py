"""YOLOX baseline: COCO tile loading, config, accumulation / lr / early-stop math (no torch needed),
plus ONNX parity vs YOLOX's PyTorch inference where torch, a trained run and val tiles exist."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import train_yolox

from colonycount._env import env_path


def test_load_split_labels_are_class0_centre_size_pixels(tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 640, "height": 320},
            {"id": 2, "file_name": "empty.jpg", "width": 640, "height": 640},
        ],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [64, 32, 128, 64]}],
        "categories": [{"id": 1, "name": "colony"}],
    }
    (tmp_path / "_annotations.coco.json").write_text(json.dumps(coco))
    items, gt = train_yolox.load_split(tmp_path)
    assert gt == coco
    (pa, ia, la), (pe, ie, le) = items
    assert (pa, ia, pe, ie) == (tmp_path / "a.jpg", 1, tmp_path / "empty.jpg", 2)
    assert la.tolist() == [[0, 128, 64, 128, 64]] and le.shape == (0, 5)


def test_tiles_pads_bottom_right_and_collate_pads_labels(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("torch")
    cv2.imwrite(str(tmp_path / "a.png"), np.full((320, 640, 3), 7, np.uint8))
    lab = np.array([[0, 100, 50, 20, 10]], np.float32)
    ds = train_yolox.Tiles(
        [(tmp_path / "a.png", 1, lab), (tmp_path / "a.png", 2, lab[:0])], 640, False
    )
    img, labels, image_id = ds[0]
    assert img.shape == (3, 640, 640) and img[:, 319].max() == 7 and img[:, 320:].max() == 0
    imgs, targets, ids = train_yolox.collate([ds[0], ds[1]])
    assert tuple(imgs.shape) == (2, 3, 640, 640) and tuple(targets.shape) == (2, 1, 5)
    assert targets[0].tolist() == lab.tolist() and targets[1].sum() == 0 and ids == [1, 2]


def _gradient_tile(tmp_path):
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((320, 640, 3), np.uint8)
    img[..., 0] = np.arange(640) % 256  # left/right and top/bottom tell apart
    img[..., 1] = (np.arange(320) % 256)[:, None]
    cv2.imwrite(str(tmp_path / "a.png"), img)
    return img, np.array([[0, 100, 50, 20, 10]], np.float32)


def test_augment_flips_horizontally_only(tmp_path, monkeypatch):
    img, lab = _gradient_tile(tmp_path)
    monkeypatch.setattr(train_yolox, "HSV_PROB", 0.0)  # isolate the flip
    monkeypatch.setattr(train_yolox.random, "random", lambda: 0.0)  # every flip that exists
    out, labels, _ = train_yolox.Tiles([(tmp_path / "a.png", 1, lab)], 640, True)[0]
    assert labels.tolist() == [[0, 540, 50, 20, 10]] and lab[0, 1] == 100  # input untouched
    assert (out[:, :320, :640] == img[:, ::-1].transpose(2, 0, 1)).all()  # no vertical flip
    monkeypatch.setattr(train_yolox.random, "random", lambda: 0.99)  # 0.99 >= flip_prob
    out, labels, _ = train_yolox.Tiles([(tmp_path / "a.png", 1, lab)], 640, True)[0]
    assert labels.tolist() == lab.tolist() and (out[:, :320] == img.transpose(2, 0, 1)).all()


def test_augment_hsv_keeps_shape_dtype_and_boxes(tmp_path, monkeypatch):
    img, lab = _gradient_tile(tmp_path)
    monkeypatch.setattr(train_yolox, "FLIP_PROB", 0.0)  # isolate HSV
    np.random.seed(1)
    out, labels, _ = train_yolox.Tiles([(tmp_path / "a.png", 1, lab)], 640, True)[0]
    assert out.shape == (3, 640, 640) and out.dtype == np.float32
    assert labels.tolist() == lab.tolist() and out[:, 320:].max() == 0
    assert (out[:, :320, :640] != img.transpose(2, 0, 1)).any()  # seed 1 changes h, s or v

    from third_party.yolox.data_augment import augment_hsv  # YOLOX's own gains, as TrainTransform

    assert augment_hsv.__defaults__ == train_yolox.HSV_GAINS == (5, 30, 30)


def test_mosaic_epochs_follow_no_aug_epochs():
    assert train_yolox.NO_AUG_EPOCHS == 15  # YOLOX Exp defaults
    assert (train_yolox.FLIP_PROB, train_yolox.HSV_PROB) == (0.5, 1.0)
    assert train_yolox.mosaic_epochs(12) == train_yolox.mosaic_epochs(15) == 0
    with pytest.raises(ValueError, match="mosaic"):
        train_yolox.mosaic_epochs(20)
    c = dict(
        epochs=12, imgsz=640, seed=0, patience=5, nbs=16, batch=2, accumulate=8, tiles="d/tiles_640"
    )
    r = train_yolox.comparable(c, amp=False)["recipe"]
    assert (r["mosaic_epochs"], r["no_aug_epochs"], r["hsv_gains"]) == (0, 15, [5, 30, 30])
    assert train_yolox.comparable(c, amp=False)["tiles"] == "tiles_640"


def test_config_effective_batch_and_resume_args(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYCOUNT_SEED", "7")
    for name in ("TRAIN_BATCH_SIZE", "TRAIN_GRAD_ACCUM", "TRAIN_TILE_SIZE", "TRAIN_EPOCHS"):
        monkeypatch.delenv(f"COLONYCOUNT_{name}", raising=False)
    assert train_yolox.parse_args(["--size", "m", "--data-dir", str(tmp_path)]).epochs == 12
    with pytest.raises(SystemExit):  # > 15 epochs would need YOLOX's mosaic phase
        train_yolox.parse_args(["--size", "m", "--data-dir", str(tmp_path), "--epochs", "20"])
    args = train_yolox.parse_args(["--size", "m", "--data-dir", str(tmp_path), "--epochs", "12"])
    c = train_yolox.build_config(args)
    assert (c["epochs"], c["imgsz"], c["seed"], c["patience"]) == (12, 640, 7, 5)
    assert (c["nbs"], c["batch"], c["accumulate"]) == (16, 2, 8)  # RF-DETR's 2 x 8
    assert c["lr"] == pytest.approx(0.01 / 64 * 16)
    assert Path(c["out"]).parent == tmp_path / "runs" / "yolox_m"
    assert c["tiles"] == str(tmp_path / "public" / "tiles_640")
    json.dumps(c)  # stored in checkpoints loaded with weights_only=True

    monkeypatch.setenv("COLONYCOUNT_TRAIN_EPOCHS", "12")
    c = train_yolox.build_config(train_yolox.parse_args(["--size", "tiny", "--batch", "8"]))
    assert (c["epochs"], c["nbs"], c["batch"], c["accumulate"]) == (12, 16, 8, 2)
    assert [train_yolox.accumulation(16, b) for b in (1, 3, 6, 16, 32)] == [16, 5, 3, 1, 1]

    with pytest.raises(SystemExit):
        train_yolox.parse_args(["--size", "tiny", "--resume", str(tmp_path / "nope.pth")])
    with pytest.raises(SystemExit):
        train_yolox.parse_args(["--size", "s"])


def test_comparable_warns_without_accumulation(capsys):
    c = dict(
        epochs=12,
        imgsz=640,
        seed=0,
        patience=5,
        nbs=16,
        batch=32,
        accumulate=1,
        tiles="d/tiles_640",
    )
    assert train_yolox.comparable(c, amp=True)["optimizer"] == "SGD"
    assert "WARNING" in capsys.readouterr().out


def test_lr_warmup_then_cosine():
    lr = 0.0025
    assert 0 < train_yolox.lr_at(0, 100, 10, lr) < lr / 50  # quadratic, never 0
    assert train_yolox.lr_at(9, 100, 10, lr) == pytest.approx(lr)
    assert train_yolox.lr_at(10, 100, 10, lr) == pytest.approx(lr)
    assert train_yolox.lr_at(100, 100, 10, lr) == pytest.approx(lr * train_yolox.MIN_LR_RATIO)
    steps = [train_yolox.lr_at(s, 100, 10, lr) for s in range(10, 101)]
    assert steps == sorted(steps, reverse=True)


def test_early_stopping_patience():
    assert not train_yolox.should_stop([0.1, 0.2, 0.2, 0.2, 0.2, 0.2])  # 4 epochs after best
    assert train_yolox.should_stop([0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.19])  # 5: stop
    assert not train_yolox.should_stop([0.3, 0.1, 0.1, 0.1, 0.1, 0.31])


data = env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
runs = sorted((data / "runs").glob("yolox_*/*/weights/best.pth"))
weights = Path(os.environ.get("COLONYCOUNT_YOLOX_WEIGHTS", runs[-1] if runs else "none"))
tiles = data / "public/tiles_640/valid"


@pytest.mark.skipif(not weights.is_file() or not tiles.is_dir(), reason="no YOLOX run or val tiles")
def test_yolox_onnx_matches_pytorch(tmp_path):
    pytest.importorskip("torch")
    import export_yolox_onnx
    import onnxruntime as ort
    from export_yolo_onnx import val_tiles

    out = tmp_path / "m.onnx"
    net, tile = export_yolox_onnx.export(weights, out)
    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert [(i.name, i.shape) for i in s.get_inputs()] == [("images", [1, 3, tile, tile])]
    assert [o.name for o in s.get_outputs()] == ["boxes", "scores"]
    p = export_yolox_onnx.parity(net, tile, out, val_tiles(tiles, 4, tile))
    assert p["ok"] and p["n_boxes"] > 0, p
