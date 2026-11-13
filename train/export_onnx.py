"""Export a trained RF-DETR checkpoint to ONNX following colonycount's model contract (task 2.5).

Usage: uv run --no-sync python train/export_onnx.py --checkpoint <run dir>/checkpoint_best_total.pth
       [--out FILE.onnx] [--tiles DIR] [--n-tiles 8]

The graph takes ``images`` float32 [1, 3, T, T] RGB in 0..1 (T = the checkpoint's training
resolution) and returns ``boxes`` [N, 4] xyxy tile pixels and ``scores`` [N], best first, N =
rfdetr's num_select. ImageNet normalisation and the postprocess (sigmoid of the colony logit, top-k,
cxcywh -> xyxy pixels, clamp to the tile) live inside the graph. The no-object logit slot is
dropped. Writes FILE.onnx, FILE.onnx.sha256, then checks ONNX Runtime CPU against rfdetr's own
predict() on real val tiles.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from colonycount._env import env_path

PARITY_THRESHOLD = 0.3
SCORE_TOL = 1e-4  # absolute, scores in 0..1
BOX_TOL = 0.01  # pixels


def export(checkpoint: Path, out: Path):
    """Write the ONNX file and its .sha256; return the loaded rfdetr model for parity checks."""
    import torch
    from rfdetr import RFDETR
    from rfdetr.export._backend import _switch_to_export_mode  # rfdetr is pinned in the extra
    from rfdetr.export.prepare import prepare_export_graph

    rf = RFDETR.from_checkpoint(checkpoint, device="cpu")
    if len(rf.class_names) != 1:
        raise SystemExit(f"expected a 1-class model, got {rf.class_names}")
    tile = rf.model_config.resolution  # trained on native-resolution tiles of this size
    net = deepcopy(rf.model.model).cpu().eval()
    # freezes the backbone's position embeddings at T x T and runs one sanity pass
    prepare_export_graph(net, rf.model_config, shape=(tile, tile), device="cpu")
    _switch_to_export_mode(net)  # forward(normalised images) -> (cxcywh 0..1 boxes, logits)

    class Contract(torch.nn.Module):
        def __init__(self, net, k):
            super().__init__()
            self.net, self.k = net, k
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, images):
            cxcywh, logits = self.net((images - self.mean) / self.std)
            scores, idx = logits[0, :, 0].sigmoid().topk(self.k)  # slot 0 = colony
            c = cxcywh[0, idx] * tile
            boxes = torch.cat([c[:, :2] - c[:, 2:] / 2, c[:, :2] + c[:, 2:] / 2], 1)
            return boxes.clamp(0, tile), scores

    k = min(rf.model_config.num_select, rf.model_config.num_queries)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Contract(net, k).eval(),
        (torch.rand(1, 3, tile, tile),),
        str(out),
        input_names=["images"],
        output_names=["boxes", "scores"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # rfdetr's own exporter uses the legacy tracer too
    )
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    return rf


def parity(rf, onnx_path: Path, tiles: list[Path], threshold: float = PARITY_THRESHOLD) -> dict:
    """Compare ONNX Runtime CPU vs rfdetr predict() (PyTorch, CPU) above `threshold` on tiles;
    max abs differences of matched (score-sorted) boxes and scores, and "ok" within tolerance."""
    import cv2

    from colonycount.detection.runner import infer, load_session

    session = load_session(str(onnx_path))
    worst = {"n_tiles": len(tiles), "n_boxes": 0, "count_mismatch": 0, "score": 0.0, "box": 0.0}
    for t in tiles:
        rgb = cv2.cvtColor(cv2.imread(str(t)), cv2.COLOR_BGR2RGB)
        det = rf.predict(rgb, threshold=threshold, include_source_image=False)
        ref = det.class_id == 0
        ref_boxes, ref_scores = det.xyxy[ref], det.confidence[ref]
        boxes, scores = infer(session, rgb, rf.model_config.resolution)
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


def val_tiles(tiles_dir: Path, n: int) -> list[Path]:
    """The n val tiles with the most boxes (deterministic), so parity sees crowded tiles too."""
    import json

    coco = json.loads((tiles_dir / "_annotations.coco.json").read_text())
    per = {i["id"]: 0 for i in coco["images"]}
    for a in coco["annotations"]:
        per[a["image_id"]] += 1
    names = {i["id"]: i["file_name"] for i in coco["images"]}
    ids = sorted(per, key=lambda i: (-per[i], names[i]))
    step = max(1, len(ids) // n)
    return [tiles_dir / names[i] for i in ids[::step][:n]]


def main() -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default: <data dir>/models/public-rfdetr-nano-<date>")
    ap.add_argument("--tiles", type=Path, help="val tiles (default: tiles_<T>/valid of the data)")
    ap.add_argument("--n-tiles", type=int, default=8)
    args = ap.parse_args()
    data = env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
    args.out = args.out or data / "models" / f"public-rfdetr-nano-{time.strftime('%Y%m%d')}.onnx"

    rf = export(args.checkpoint, args.out)
    print(f"wrote {args.out}")
    print(args.out.with_name(args.out.name + ".sha256").read_text().strip())
    tiles = args.tiles or data / "public" / f"tiles_{rf.model_config.resolution}" / "valid"
    p = parity(rf, args.out, val_tiles(tiles, args.n_tiles))
    print(f"parity vs rfdetr predict() at score > {PARITY_THRESHOLD}: {p}")
    if not p["ok"]:
        raise SystemExit(f"parity failed (tolerance: score {SCORE_TOL}, box {BOX_TOL} px)")


if __name__ == "__main__":
    main()
