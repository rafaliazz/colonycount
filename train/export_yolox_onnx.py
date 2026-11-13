"""Export a trained YOLOX baseline (train_yolox.py) to ONNX per colonycount's model contract.

Usage: uv run --no-sync python train/export_yolox_onnx.py --weights <run dir>/weights/best.pth
       [--out FILE.onnx] [--tiles DIR] [--n-tiles 8]

The graph takes ``images`` float32 [1, 3, T, T] RGB 0..1 (T = the training imgsz), turns it into
the BGR 0..255 YOLOX was trained on, and returns ``boxes`` [N, 4] xyxy tile pixels and ``scores``
[N] (objectness x class score), best first: the EMA weights' decoded head output above SCORE_FLOOR
goes through the NMS of YOLOX's own postprocess() (IoU 0.45, its demo default), up to 300 boxes,
clipped to the tile. Writes FILE.onnx and FILE.onnx.sha256, then checks ONNX Runtime CPU against
YOLOX's PyTorch inference + postprocess() on full-size val tiles.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from export_yolo_onnx import val_tiles

from colonycount._env import env_path

SCORE_FLOOR = 0.1  # below evaluate.py's lowest swept threshold (0.2); keeps in-graph NMS cheap
NMS_IOU = 0.45  # YOLOX postprocess() / demo default
MAX_DET = 300
PARITY_THRESHOLD = 0.3
SCORE_TOL = 1e-4
BOX_TOL = 0.01  # pixels


def load(weights: Path):
    """(YOLOX in eval mode with the checkpoint's EMA weights, tile size)."""
    import torch
    from train_yolox import build_model

    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    model = build_model(ckpt["size"])
    model.load_state_dict(ckpt["ema"])
    return model.eval(), int(ckpt["config"]["imgsz"])


def export(weights: Path, out: Path):
    """Write the ONNX file and its .sha256; return (PyTorch model, tile size)."""
    import torch
    import torchvision

    net, tile = load(weights)

    class Contract(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, images):
            y = self.net(images[:, [2, 1, 0]] * 255)[0]  # [A, 6]: cx, cy, w, h, obj, cls
            scores = y[:, 4] * y[:, 5]
            keep = scores > SCORE_FLOOR
            c, scores = y[keep, :4], scores[keep]
            boxes = torch.cat([c[:, :2] - c[:, 2:] / 2, c[:, :2] + c[:, 2:] / 2], 1)
            idx = torchvision.ops.nms(boxes, scores, NMS_IOU)[:MAX_DET]  # sorted by score
            return boxes[idx].clamp(0, tile), scores[idx]

    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            Contract(net).eval(),
            (torch.rand(1, 3, tile, tile),),
            str(out),
            input_names=["images"],
            output_names=["boxes", "scores"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    return net, tile


def reference(net, bgr: np.ndarray, threshold: float):
    """YOLOX's own PyTorch inference: model + postprocess() at `threshold`, clipped to the tile."""
    import torch
    from third_party.yolox import postprocess

    tile = bgr.shape[0]
    x = torch.from_numpy(bgr.transpose(2, 0, 1)[None].astype(np.float32))
    with torch.no_grad():
        d = postprocess(net(x), 1, threshold, NMS_IOU)[0]
    if d is None:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    d = d[:MAX_DET]
    return d[:, :4].clamp(0, tile).numpy(), (d[:, 4] * d[:, 5]).numpy()


def parity(net, tile: int, onnx_path: Path, tiles: list[Path], threshold=PARITY_THRESHOLD) -> dict:
    """ONNX Runtime CPU vs YOLOX PyTorch (CPU) at score >= `threshold`: max abs differences of
    score-sorted boxes and scores, and "ok" within tolerance."""
    import cv2

    from colonycount.detection.runner import infer, load_session

    session = load_session(str(onnx_path))
    worst = {"n_tiles": len(tiles), "n_boxes": 0, "count_mismatch": 0, "score": 0.0, "box": 0.0}
    for t in tiles:
        bgr = cv2.imread(str(t))
        ref_boxes, ref_scores = reference(net, bgr, threshold)
        boxes, scores = infer(session, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), tile)
        keep = scores >= threshold  # postprocess() keeps scores >= conf_thre
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes) != len(ref_boxes):
            worst["count_mismatch"] += 1
            continue
        a, b = np.argsort(-scores, kind="stable"), np.argsort(-ref_scores, kind="stable")
        worst["n_boxes"] += len(boxes)
        if len(boxes):
            worst["score"] = max(worst["score"], float(np.abs(scores[a] - ref_scores[b]).max()))
            worst["box"] = max(worst["box"], float(np.abs(boxes[a] - ref_boxes[b]).max()))
    worst["ok"] = (
        not worst["count_mismatch"] and worst["score"] <= SCORE_TOL and worst["box"] <= BOX_TOL
    )
    return worst


def main() -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="train_yolox.py's weights/best.pth")
    ap.add_argument("--out", type=Path, help="default: <data dir>/models/yolox_<size>-<date>.onnx")
    ap.add_argument("--tiles", type=Path, help="val tiles (default: tiles_<T>/valid of the data)")
    ap.add_argument("--n-tiles", type=int, default=8)
    args = ap.parse_args()
    data = env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
    run = args.weights.parents[2].name  # runs/yolox_<size>/<timestamp>/weights/best.pth
    args.out = args.out or data / "models" / f"{run}-{time.strftime('%Y%m%d')}.onnx"

    net, tile = export(args.weights, args.out)
    print(f"wrote {args.out}")
    print(args.out.with_name(args.out.name + ".sha256").read_text().strip())
    tiles = args.tiles or data / "public" / f"tiles_{tile}" / "valid"
    p = parity(net, tile, args.out, val_tiles(tiles, args.n_tiles, tile))
    print(f"parity vs YOLOX postprocess() at score >= {PARITY_THRESHOLD}: {p}")
    if not p["ok"]:
        raise SystemExit(f"parity failed (tolerance: score {SCORE_TOL}, box {BOX_TOL} px)")


if __name__ == "__main__":
    main()
