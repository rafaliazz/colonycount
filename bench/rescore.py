"""Re-score cached detections (evaluate.py --save-detections) without running a model.

Usage: uv run python bench/rescore.py CACHE.npz [--threshold T ...] [--nms-iou X]
       [--plate | --no-plate] [--plate-pad PX] [--out DIR]

Applies, to the cached raw detections, what count() does after inference: threshold, drop box
centres outside the plate circle (radius + --plate-pad), NMS. Then the same report as evaluate.py.
Default: sweep evaluate's thresholds, COLONYCOUNT_NMS_IOU, plate restriction as in the cached run.
Exact for a cache made in the same mode; --plate on a whole-image (--no-plate) cache only
approximates a crop run, because the crop run tiles a different image.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import evaluate
import numpy as np

from colonycount.detection.tiling import nms


def load(path: Path) -> tuple[dict, list[dict]]:
    """Header fields and one raw-detection record per plate."""
    z = np.load(path)
    head = {k: z[k].item() for k in ("model", "sha256", "split", "no_plate")}
    plates = []
    for i, f in enumerate(z["names"]):
        a, b = z["raw_offsets"][i : i + 2]
        c, d = z["gt_offsets"][i : i + 2]
        plates.append(
            {
                "file": str(f),
                "source": str(z["source"][i]),
                "boxes": z["boxes"][a:b],
                "scores": z["scores"][a:b],
                "circle": None if np.isnan(z["circle"][i, 0]) else tuple(z["circle"][i]),
                "gt_boxes": z["gt_boxes"][c:d],
                "seconds": float(z["seconds"][i]),
            }
        )
    return head, plates


def rescore(raw: list[dict], threshold: float, use_plate: bool, pad: float = 0.0) -> list[dict]:
    """evaluate.py plate dicts for the detections at or above ``threshold``."""
    plates = []
    for p in raw:
        gt = p["gt_boxes"]
        gt_xy = gt[:, :2] + gt[:, 2:] / 2
        circle = p["circle"] if use_plate else None
        boxes, scores = p["boxes"], p["scores"]
        keep = scores >= threshold
        centres = (boxes[:, :2] + boxes[:, 2:]) / 2
        outside = 0
        if circle:
            cx, cy, r = circle
            keep &= np.hypot(centres[:, 0] - cx, centres[:, 1] - cy) <= r + pad
            outside = int((np.hypot(*(gt_xy - circle[:2]).T) > r).sum())
        boxes, scores, centres = boxes[keep], scores[keep], centres[keep]
        if len(boxes):
            idx = nms(boxes, scores)
            scores, centres = scores[idx], centres[idx]
        plates.append(
            {
                "file": p["file"],
                "source": p["source"],
                "gt": len(gt),
                "gt_xy": gt_xy,
                "gt_r": gt[:, 2:].mean(axis=1) / 2,
                "gt_outside": outside,
                "plate_found": circle is not None,
                "conf": [float(s) for s in scores],
                "xy": [(float(x), float(y)) for x, y in centres],
            }
        )
    return plates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path, help="detections .npz from evaluate.py --save-detections")
    ap.add_argument("--threshold", type=float, nargs="+", help="default: evaluate's sweep")
    ap.add_argument("--nms-iou", type=float, help="default: COLONYCOUNT_NMS_IOU or 0.5")
    ap.add_argument("--plate", action=argparse.BooleanOptionalAction, help="default: as cached")
    ap.add_argument("--plate-pad", type=float, default=0.0, help="px added to the circle radius")
    ap.add_argument("--out", type=Path, help="also write <out>/<cache stem>-rescore.json/.md")
    args = ap.parse_args()
    if args.nms_iou is not None:
        os.environ["COLONYCOUNT_NMS_IOU"] = str(args.nms_iou)
    head, raw = load(args.cache)
    use_plate = (not head["no_plate"]) if args.plate is None else args.plate
    thresholds = args.threshold or evaluate.THRESHOLDS
    plates = rescore(raw, min(thresholds), use_plate, args.plate_pad)
    rep = evaluate.report(plates, head["split"], not use_plate, head["model"], thresholds)
    md = evaluate.markdown(rep, [p["seconds"] for p in raw])
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        stem = args.out / (args.cache.stem + "-rescore")
        stem.with_suffix(".json").write_text(json.dumps(rep, indent=1) + "\n")
        stem.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
