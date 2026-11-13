"""OpenCFU baseline (model-benchmark task 2.1), scored with bench/evaluate.py's metrics.

Usage: uv run python bench/opencfu_runner.py [--split val] [--no-plate] [--tune] [--out DIR]
Needs Docker and the image: ``docker build -t colonycount-opencfu bench/opencfu``.

Each plate is cropped like colonycount does (find_plate's bounding square; whole image when no
plate is found, or always with --no-plate), resized so its longest side is ``side`` pixels and
written to a temporary folder.
One ``docker run`` per setting runs OpenCFU on every image (folder mounted
read-only); its per-colony CSV (X, Y, Radius) is mapped back to image pixels, colonies outside the
plate circle are dropped (as count() does), and the plates go through evaluate.report at a single
"threshold" (every OpenCFU colony has confidence 1). --tune runs the grid below on val and keeps the
setting with the lowest MdAPE on 30-300 plates (tie-break MAE); without it the given flags are used.
Writes <out>/opencfu-<split>[-noplate].json/.md (and the grid with --tune).
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from evaluate import ground_truth, markdown, report, summarize

from colonycount._env import env_path
from colonycount.detection.pipeline import load_image
from colonycount.detection.plate import find_plate

IMAGE = "colonycount-opencfu"
# Tuning grid. Radii are fractions of `side` (the resized image's longest side), so they scale
# with the plate rather than the camera's resolution.
GRID = {
    "side": [1000, 1500],
    "mode": ["reg -t 10", "inv -t 5", "inv -t 10", "bil -t 3", "bil -t 5", "bil -t 10", "auto"],
    "r_min": [0.002, 0.004],
    "r_max": [0.03, 0.06],
}
DEFAULT = {"side": 1000, "mode": "reg -t 10", "r_min": 0.003, "r_max": 0.05}


def flags(setting: dict) -> str:
    """OpenCFU command-line flags for a setting (radii in pixels of the resized image)."""
    mode = "-a" if setting["mode"] == "auto" else "-d " + setting["mode"]
    r = max(1, round(setting["r_min"] * setting["side"]))
    big = max(r + 1, round(setting["r_max"] * setting["side"]))
    return f"{mode} -r {r} -R {big}"


def parse(text: str) -> np.ndarray:
    """OpenCFU's CSV on stdout -> Nx3 array of valid colonies (x, y, radius), resized pixels."""
    rows = [
        (float(r["X"]), float(r["Y"]), float(r["Radius"]))
        for r in csv.DictReader(io.StringIO(text))
        if r.get("IsValid") == "1"
    ]
    return np.asarray(rows, float).reshape(-1, 3)


def to_plate(dets: np.ndarray, prep: dict) -> dict:
    """Map colonies back to image pixels, drop those outside the plate circle -> xy and sizes."""
    xy = dets[:, :2] / prep["scale"] + prep["offset"]
    d = 2 * dets[:, 2] / prep["scale"]
    if prep["circle"]:
        cx, cy, r = prep["circle"]
        keep = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= r
        xy, d = xy[keep], d[keep]
    return {"xy": [tuple(p) for p in xy], "diameter": d.tolist(), "conf": [1.0] * len(xy)}


def prepare(data_dir: Path, split: str, no_plate: bool, sides: list[int], tmp: Path) -> list[dict]:
    """Crop + resize every plate once per side; returns GT and geometry per plate."""
    plates = []
    for k, (im, boxes) in enumerate(ground_truth(data_dir, split)):
        img = load_image(data_dir / im["file_name"])
        h, w = img.shape[:2]
        circle = None if no_plate else find_plate(img)
        x0, y0, x1, y1 = 0, 0, w, h
        if circle:  # the same bounding square as colonycount.count
            cx, cy, r = circle
            x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
            x1, y1 = min(w, int(np.ceil(cx + r)) + 1), min(h, int(np.ceil(cy + r)) + 1)
        crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
        gt_xy = boxes[:, :2] + boxes[:, 2:] / 2
        outside = int((np.hypot(*(gt_xy - circle[:2]).T) > circle[2]).sum()) if circle else 0
        scales = {}
        for side in sides:
            s = side / max(crop.shape[:2])
            out = tmp / str(side) / f"{k:05d}.png"
            out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out), cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            scales[side] = s
        plates.append(
            {
                "file": im["file_name"],
                "source": im["source"],
                "gt": len(boxes),
                "gt_xy": gt_xy,
                "gt_r": boxes[:, 2:].mean(axis=1) / 2,
                "gt_outside": outside,
                "plate_found": circle is not None,
                "png": f"{k:05d}.png",
                "offset": np.array([x0, y0], float),
                "circle": circle,
                "scales": scales,
            }
        )
    return plates


def run_opencfu(folder: Path, args: str, jobs: int) -> dict[str, tuple[str, float]]:
    """OpenCFU on every PNG in folder, in one container -> {name: (csv text, seconds)}."""
    with tempfile.TemporaryDirectory() as out:
        script = (
            "ls /in | xargs -P $JOBS -I{} sh -c 's=$(date +%s%N); "
            "opencfu $ARGS -i /in/{} > /out/{}.csv 2>/dev/null; "
            "echo $(( $(date +%s%N) - s )) > /out/{}.ns'"
        )
        subprocess.run(
            ["docker", "run", "--rm", "--name", f"colonycount-opencfu-{os.getpid()}"]
            + ["--network", "none", "-e", f"ARGS={args}", "-e", f"JOBS={jobs}"]
            + ["-v", f"{folder}:/in:ro", "-v", f"{out}:/out", IMAGE, "sh", "-c", script],
            check=True,
        )
        return {
            f.stem: (f.read_text(), int((Path(out) / f"{f.stem}.ns").read_text()) / 1e9)
            for f in Path(out).glob("*.csv")
        }


def evaluate_setting(plates, tmp, setting, jobs) -> tuple[list[dict], list[float]]:
    side = setting["side"]
    res = run_opencfu(tmp / str(side), flags(setting), jobs)
    scored, seconds = [], []
    for p in plates:
        text, sec = res[p["png"]]
        prep = {"scale": p["scales"][side], "offset": p["offset"], "circle": p["circle"]}
        scored.append(p | to_plate(parse(text), prep))
        seconds.append(sec)
    return scored, seconds


def main() -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true", help="test is for the final choice only")
    ap.add_argument("--no-plate", action="store_true", help="whole image, no plate crop")
    ap.add_argument("--tune", action="store_true", help="grid-search the settings (val only)")
    ap.add_argument("--setting", type=json.loads, default=DEFAULT, help="JSON, keys as DEFAULT")
    ap.add_argument("--jobs", type=int, default=6, help="parallel OpenCFU processes when tuning")
    ap.add_argument("--data-dir", type=Path, help="default: COLONYCOUNT_DATA_DIR")
    ap.add_argument("--out", type=Path, help="default: <data dir>/bench")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        ap.error("the test split is reserved for the final model choice; pass --allow-test")
    if args.tune and args.split != "val":
        ap.error("settings are tuned on val only")
    args.data_dir = args.data_dir or env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
    args.out = args.out or args.data_dir / "bench"
    stem = args.out / f"opencfu-{args.split}{'-noplate' if args.no_plate else ''}"
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sides = GRID["side"] if args.tune else [args.setting["side"]]
        plates = prepare(args.data_dir, args.split, args.no_plate, sides, tmp)
        grid, best = [], args.setting
        if args.tune:
            for values in itertools.product(*GRID.values()):
                setting = dict(zip(GRID, values, strict=True))
                scored, _ = evaluate_setting(plates, tmp, setting, args.jobs)
                row = setting | {"flags": flags(setting)} | summarize(scored, 1.0)
                grid.append(row)
                print(json.dumps(row), flush=True)
            best = min(grid, key=lambda r: (r["mdape_30_300"] or float("inf"), r["mae"]))
            best = {k: best[k] for k in GRID}
            stem.with_name(stem.name + "-grid.json").write_text(json.dumps(grid, indent=1) + "\n")
        # one process at a time, so seconds per plate are comparable with the model's CPU time
        scored, seconds = evaluate_setting(plates, tmp, best, jobs=1)

    name = f"OpenCFU 4.0.0 ({flags(best)}, longest side {best['side']} px)"
    rep = report(scored, args.split, args.no_plate, name, thresholds=[1.0])
    rep |= {"setting": best, "flags": flags(best), "threshold_curve": None}
    stem.with_suffix(".json").write_text(json.dumps(rep, indent=1) + "\n")
    md = markdown(rep, seconds).replace(", threshold 1.0", "", 1)
    if grid:
        md += "\n" + grid_table(grid)
    stem.with_suffix(".md").write_text(md)
    print(md)


def grid_table(grid: list[dict]) -> str:
    cols = ["side", "mode", "r_min", "r_max", "flags", "mae", "bias", "mdape_30_300"]
    cols += ["precision", "recall"]
    lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for r in sorted(grid, key=lambda r: (r["mdape_30_300"] or float("inf"), r["mae"])):
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
