"""Count-accuracy benchmark: colonycount's real pipeline on whole plates of a COCO split (task 2.3).

Usage: uv run python bench/evaluate.py --model FILE.onnx [--split val] [--no-plate] [--out DIR]
       [--limit N]

Runs ``colonycount.count`` (plate finding, tiling, seam-aware NMS, threshold) once per plate at the
lowest swept threshold; higher thresholds keep the detections scoring at or above them. That is
exactly what count() would return at that threshold, because greedy NMS only ever lets a
higher-scoring box suppress a lower one. Ground truth per plate = its annotation count.

Metrics, per source and overall: MAE, bias (mean of predicted minus true), MdAPE on plates whose
true count is 30-300 (and on all plates with a non-zero count), detection precision/recall (a
prediction matches a true colony when its centre lies within half the box's mean side of the box
centre; greedy one-to-one, nearest first), plate_found rate, and errors by density bucket. The
threshold is picked on this split: lowest MdAPE (30-300), tie-break MAE. The test split is reserved
for the final model choice and needs --allow-test. Writes <out>/<split>[-noplate].json
(deterministic, no timings) and .md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from unittest import mock

import numpy as np
from dotenv import load_dotenv

from colonycount._env import env_path
from colonycount.detection import pipeline
from colonycount.detection.pipeline import count, load_image
from colonycount.detection.plate import find_plate
from colonycount.detection.tiling import detect_tiled

THRESHOLDS = [round(0.2 + 0.05 * i, 2) for i in range(13)]  # 0.20 .. 0.80
RANGE = (30, 300)
MIN_SCORE = 0.05  # raw detections below this are not cached
BUCKETS = {"<30": (0, 29), "30-300": RANGE, ">300": (301, float("inf"))}


def mdape(pred, gt) -> float | None:
    """Median absolute percentage error over plates with a non-zero true count."""
    apes = [abs(p - g) / g * 100 for p, g in zip(pred, gt, strict=True) if g > 0]
    return statistics.median(apes) if apes else None


def match(pred_xy: np.ndarray, gt_xy: np.ndarray, gt_radius: np.ndarray) -> int:
    """True positives: greedy one-to-one matching, nearest pairs first, within each GT radius."""
    if not len(pred_xy) or not len(gt_xy):
        return 0
    d = np.hypot(*(pred_xy[:, None, :] - gt_xy[None, :, :]).transpose(2, 0, 1))
    p, g = np.nonzero(d <= gt_radius[None, :])
    used_p, used_g = set(), set()
    for k in np.lexsort((g, p, d[p, g])):  # by distance, ties by index: deterministic
        if p[k] not in used_p and g[k] not in used_g:
            used_p.add(p[k])
            used_g.add(g[k])
    return len(used_p)


def summarize(plates: list[dict], t: float) -> dict:
    """Metrics at threshold t over the given plates."""
    gt = [p["gt"] for p in plates]
    pred = [sum(c >= t for c in p["conf"]) for p in plates]
    err = [a - b for a, b in zip(pred, gt, strict=True)]
    tp = fp = fn = 0
    for p, n in zip(plates, pred, strict=True):
        keep = np.asarray(p["conf"]) >= t
        m = match(np.asarray(p["xy"]).reshape(-1, 2)[keep], p["gt_xy"], p["gt_r"])
        tp, fp, fn = tp + m, fp + n - m, fn + p["gt"] - m
    in_range = [i for i, g in enumerate(gt) if RANGE[0] <= g <= RANGE[1]]
    md = mdape([pred[i] for i in in_range], [gt[i] for i in in_range])
    md_all = mdape(pred, gt)
    return {
        "n": len(plates),
        "mae": round(float(np.mean(np.abs(err))), 3) if plates else None,
        "bias": round(float(np.mean(err)), 3) if plates else None,
        "mdape_30_300": None if md is None else round(md, 3),
        "n_30_300": len(in_range),
        "mdape_all": None if md_all is None else round(md_all, 3),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "plate_found": round(sum(p["plate_found"] for p in plates) / len(plates), 4)
        if plates
        else None,
    }


def report(
    plates: list[dict], split: str, no_plate: bool, model: str, thresholds=THRESHOLDS
) -> dict:
    curve = {str(t): summarize(plates, t) for t in thresholds}
    best = min(
        thresholds,
        key=lambda t: (curve[str(t)]["mdape_30_300"] or float("inf"), curve[str(t)]["mae"]),
    )
    sources = sorted({p["source"] for p in plates})
    rows = {s: summarize([p for p in plates if p["source"] == s], best) for s in sources}
    rows["overall"] = curve[str(best)]
    buckets = {
        s: {
            b: summarize(
                [p for p in plates if lo <= p["gt"] <= hi and s in (p["source"], "all")], best
            )
            for b, (lo, hi) in BUCKETS.items()
        }
        for s in [*sources, "all"]
    }
    return {
        "model": model,
        "split": split,
        "plate_restriction": not no_plate,
        "match_radius": "0.5 x mean GT box side, around the GT box centre",
        "threshold": best,
        "by_source": rows,
        "by_density": buckets,
        "threshold_curve": curve,
        "plate_not_found": sorted(p["file"] for p in plates if not p["plate_found"]),
        "gt_outside_plate": {
            s: round(
                sum(p["gt_outside"] for p in plates if s in (p["source"], "all"))
                / max(1, sum(p["gt"] for p in plates if s in (p["source"], "all"))),
                4,
            )
            for s in [*sources, "all"]
        },
        "plates": [
            {k: p[k] for k in ("file", "source", "gt", "plate_found")}
            | {"pred": sum(c >= best for c in p["conf"])}
            for p in plates
        ],
    }


def markdown(rep: dict, seconds: list[float]) -> str:
    def table(rows: dict, first: str) -> list[str]:
        cols = [
            "n",
            "mae",
            "bias",
            "mdape_30_300",
            "n_30_300",
            "mdape_all",
            "precision",
            "recall",
            "plate_found",
        ]
        lines = [f"| {first} | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        return lines + [
            f"| {k} | " + " | ".join("-" if v[c] is None else str(v[c]) for c in cols) + " |"
            for k, v in rows.items()
        ]

    out = [
        f"## {rep['model']} on {rep['split']} (plate restriction "
        f"{'on' if rep['plate_restriction'] else 'off'}), threshold {rep['threshold']}",
        "",
        *table(rep["by_source"], "source"),
        "",
    ]
    for s, b in rep["by_density"].items():
        out += [*table(b, f"{s} by GT count"), ""]
    if rep["threshold_curve"]:
        out += [*table(rep["threshold_curve"], "threshold (all)"), ""]
    out += [
        f"plate not found: {len(rep['plate_not_found'])} of {len(rep['plates'])} plates; "
        f"GT colonies outside the found plate circle: {rep['gt_outside_plate']}",
        f"CPU time per plate: median {statistics.median(seconds):.2f} s, "
        f"mean {statistics.mean(seconds):.2f} s, max {max(seconds):.2f} s",
    ]
    return "\n".join(out) + "\n"


def ground_truth(data_dir: Path, split: str):
    """Yield (COCO image record, its GT boxes as an Nx4 xywh array), sorted by file name."""
    coco = json.loads((data_dir / "public" / "annotations" / f"{split}.json").read_text())
    anns: dict[int, list] = {i["id"]: [] for i in coco["images"]}
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a["bbox"])
    for im in sorted(coco["images"], key=lambda i: i["file_name"]):
        yield im, np.asarray(anns[im["id"]], float).reshape(-1, 4)


def run(
    model: Path, split: str, data_dir: Path, no_plate: bool, limit: int | None = None
) -> tuple[list[dict], list[float]]:
    plates, seconds = [], []
    items = list(ground_truth(data_dir, split))
    if limit:  # smoke runs: evenly spaced plates
        items = items[:: max(1, len(items) // limit)][:limit]
    for im, boxes in items:
        img = load_image(data_dir / im["file_name"])
        gt_xy = boxes[:, :2] + boxes[:, 2:] / 2
        found = find_plate(img)
        circle = None if no_plate else found
        outside = 0
        if circle:
            outside = int((np.hypot(*(gt_xy - circle[:2]).T) > circle[2]).sum())
        raw = {}

        def grab(*a, raw=raw, **k):
            raw["boxes"], raw["scores"] = out = detect_tiled(*a, **k)
            return out

        t0 = time.perf_counter()
        # count() shifts the boxes to full-image pixels in place, so raw["boxes"] ends up there.
        with mock.patch.object(pipeline, "detect_tiled", grab):
            if no_plate:
                with mock.patch.object(pipeline, "find_plate", return_value=None):
                    res = count(img, model_path=model, threshold=THRESHOLDS[0])
            else:
                res = count(img, model_path=model, threshold=THRESHOLDS[0])
        seconds.append(time.perf_counter() - t0)
        hi = raw["scores"] >= MIN_SCORE
        plates.append(
            {
                "file": im["file_name"],
                "source": im["source"],
                "gt": len(boxes),
                "gt_xy": gt_xy,
                "gt_r": boxes[:, 2:].mean(axis=1) / 2,
                "gt_outside": outside,
                "plate_found": res.plate_found,
                "conf": [d.confidence for d in res.detections],
                "xy": [(d.x, d.y) for d in res.detections],
                "raw": (raw["boxes"][hi], raw["scores"][hi]),
                "found": found,
                "size": img.shape[:2],
                "gt_boxes": boxes,
            }
        )
        print(
            f"{im['file_name']}: gt {len(boxes)} pred@{THRESHOLDS[0]} {res.total} "
            f"plate {res.plate_found} {seconds[-1]:.1f}s",
            flush=True,
        )
    return plates, seconds


def save_detections(path: Path, plates, seconds, model: Path, split: str, no_plate: bool) -> None:
    """Raw detections per plate (post seam ownership, pre threshold/plate/NMS) for rescore.py."""
    np.savez_compressed(
        path,
        model=model.name,
        sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        split=split,
        no_plate=no_plate,
        min_score=MIN_SCORE,
        names=[p["file"] for p in plates],
        source=[p["source"] for p in plates],
        seconds=seconds,
        size=[p["size"] for p in plates],
        circle=[p["found"] or (np.nan,) * 3 for p in plates],  # what the finder returned
        boxes=np.concatenate([p["raw"][0] for p in plates]),
        scores=np.concatenate([p["raw"][1] for p in plates]),
        raw_offsets=np.cumsum([0, *(len(p["raw"][1]) for p in plates)]),
        gt_boxes=np.concatenate([p["gt_boxes"] for p in plates]),
        gt_offsets=np.cumsum([0, *(len(p["gt_boxes"]) for p in plates)]),
    )


def main() -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="ONNX file")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true", help="test is for the final choice only")
    ap.add_argument("--no-plate", action="store_true", help="count the whole image, no plate crop")
    ap.add_argument("--data-dir", type=Path, help="default: COLONYCOUNT_DATA_DIR")
    ap.add_argument("--out", type=Path, help="default: <data dir>/bench")
    ap.add_argument("--save-detections", action="store_true", help="also write <stem>.npz")
    ap.add_argument("--limit", type=int, help="only N evenly spaced plates (smoke runs)")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        ap.error("the test split is reserved for the final model choice; pass --allow-test")
    args.data_dir = args.data_dir or env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
    args.out = args.out or args.data_dir / "bench"

    plates, seconds = run(args.model, args.split, args.data_dir, args.no_plate, args.limit)
    rep = report(plates, args.split, args.no_plate, args.model.name)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / (args.split + ("-noplate" if args.no_plate else ""))
    stem.with_suffix(".json").write_text(json.dumps(rep, indent=1) + "\n")
    if args.save_detections:
        save_detections(
            stem.with_suffix(".npz"), plates, seconds, args.model, args.split, args.no_plate
        )
    md = markdown(rep, seconds)
    stem.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
