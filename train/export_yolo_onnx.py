"""Export a trained YOLOv8 baseline (train_yolo.py) to ONNX per colonycount's model contract.

Usage: uv run --no-sync python train/export_yolo_onnx.py --weights <run dir>/weights/best.pt
       [--out FILE.onnx] [--tiles DIR] [--n-tiles 8]

The graph takes ``images`` float32 [1, 3, T, T] RGB 0..1 (T = the training imgsz, which is what
Ultralytics expects with no letterbox at T) and returns ``boxes`` [N, 4] xyxy tile pixels and
``scores`` [N], best first. Ultralytics' decoded head output (cxcywh pixels + sigmoid score) goes
through the same NMS as its predict() (IoU 0.7, up to 300 boxes) inside the graph, keeping scores
above SCORE_FLOOR, then clips boxes to the tile like predict(). Writes FILE.onnx and
FILE.onnx.sha256, then checks ONNX Runtime CPU against Ultralytics' own predict() on full-size val
tiles. Benchmark only: Ultralytics is AGPL-3.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from colonycount._env import env_path

SCORE_FLOOR = 0.1  # below evaluate.py's lowest swept threshold (0.2); keeps in-graph NMS cheap
NMS_IOU = 0.7  # Ultralytics predict() default
MAX_DET = 300
PARITY_THRESHOLD = 0.3
SCORE_TOL = 1e-4
BOX_TOL = 0.01  # pixels


def export(weights: Path, out: Path):
    """Write the ONNX file and its .sha256; return (ultralytics YOLO, tile size)."""
    import torch
    import torchvision
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    if len(yolo.names) != 1:
        raise SystemExit(f"expected a 1-class model, got {yolo.names}")
    tile = int(yolo.ckpt["train_args"]["imgsz"])
    net = yolo.model.float().fuse().eval()  # predict() fuses Conv+BN too

    class Contract(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, images):
            y = self.net(images)
            y = (y[0] if isinstance(y, tuple | list) else y)[0]  # [5, A]: cx, cy, w, h, score
            keep = y[4] > SCORE_FLOOR
            c, scores = y[:4, keep].T, y[4, keep]
            boxes = torch.cat([c[:, :2] - c[:, 2:] / 2, c[:, :2] + c[:, 2:] / 2], 1)
            idx = torchvision.ops.nms(boxes, scores, NMS_IOU)[:MAX_DET]  # sorted by score
            return boxes[idx].clamp(0, tile), scores[idx]  # predict() clips after NMS too

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
    return YOLO(str(weights)), tile  # fresh copy for predict()


def parity(yolo, tile: int, onnx_path: Path, tiles: list[Path], threshold=PARITY_THRESHOLD) -> dict:
    """ONNX Runtime CPU vs Ultralytics predict() (CPU) above `threshold`: max abs differences of
    score-sorted boxes and scores, and "ok" within tolerance."""
    import cv2

    from colonycount.detection.runner import infer, load_session

    session = load_session(str(onnx_path))
    worst = {"n_tiles": len(tiles), "n_boxes": 0, "count_mismatch": 0, "score": 0.0, "box": 0.0}
    for t in tiles:
        bgr = cv2.imread(str(t))
        r = yolo.predict(
            bgr, imgsz=tile, conf=threshold, iou=NMS_IOU, max_det=MAX_DET, device="cpu",
            verbose=False,
        )[0].boxes  # fmt: skip
        ref_boxes, ref_scores = r.xyxy.numpy(), r.conf.numpy()
        boxes, scores = infer(session, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), tile)
        keep = scores > threshold  # predict keeps scores strictly above
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


def val_tiles(tiles_dir: Path, n: int, tile: int) -> list[Path]:
    """n full-size val tiles spread over box counts, most first (deterministic). Full size only:
    predict() letterboxes smaller edge tiles centred, colonycount pads them bottom/right."""
    coco = json.loads((tiles_dir / "_annotations.coco.json").read_text())
    full = {i["id"]: i["file_name"] for i in coco["images"] if i["width"] == i["height"] == tile}
    per = dict.fromkeys(full, 0)
    for a in coco["annotations"]:
        if a["image_id"] in per:
            per[a["image_id"]] += 1
    ids = sorted(per, key=lambda i: (-per[i], full[i]))
    return [tiles_dir / full[i] for i in ids[:: max(1, len(ids) // n)][:n]]


def main() -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="train_yolo.py's weights/best.pt")
    ap.add_argument("--out", type=Path, help="default: <data dir>/models/yolov8<size>-<date>.onnx")
    ap.add_argument("--tiles", type=Path, help="val tiles (default: tiles_<T>/valid of the data)")
    ap.add_argument("--n-tiles", type=int, default=8)
    args = ap.parse_args()
    data = env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
    run = args.weights.parents[2].name  # runs/yolov8<size>/<timestamp>/weights/best.pt
    args.out = args.out or data / "models" / f"{run}-{time.strftime('%Y%m%d')}.onnx"

    yolo, tile = export(args.weights, args.out)
    print(f"wrote {args.out}")
    print(args.out.with_name(args.out.name + ".sha256").read_text().strip())
    tiles = args.tiles or data / "public" / f"tiles_{tile}" / "valid"
    p = parity(yolo, tile, args.out, val_tiles(tiles, args.n_tiles, tile))
    print(f"parity vs ultralytics predict() at score > {PARITY_THRESHOLD}: {p}")
    if not p["ok"]:
        raise SystemExit(f"parity failed (tolerance: score {SCORE_TOL}, box {BOX_TOL} px)")


if __name__ == "__main__":
    main()
