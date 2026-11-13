"""ONNX export vs PyTorch parity (task 2.5). Runs only where torch, rfdetr, the trained checkpoint
and the val tiles exist (COLONYCOUNT_CHECKPOINT overrides the checkpoint path); skipped in CI."""

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("rfdetr")

import export_onnx  # noqa: E402

from colonycount._env import env_path  # noqa: E402

data = env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
ckpt = Path(
    os.environ.get(
        "COLONYCOUNT_CHECKPOINT",
        data / "runs/rfdetr_nano/20261113-010505/checkpoint_best_total.pth",
    )
)
tiles = data / "public/tiles_640/valid"


@pytest.mark.skipif(not ckpt.is_file() or not tiles.is_dir(), reason="no checkpoint or val tiles")
def test_onnx_matches_pytorch(tmp_path):
    import onnxruntime as ort

    out = tmp_path / "m.onnx"
    rf = export_onnx.export(ckpt, out)
    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert [(i.name, i.shape) for i in s.get_inputs()] == [("images", [1, 3, 640, 640])]
    assert [o.name for o in s.get_outputs()] == ["boxes", "scores"]
    p = export_onnx.parity(rf, out, export_onnx.val_tiles(tiles, 4))
    assert p["ok"] and p["n_boxes"] > 0, p
