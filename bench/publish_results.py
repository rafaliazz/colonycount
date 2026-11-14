"""Copy benchmark outputs from the data folder into the repo's results/ and rebuild comparison.md.

Usage: uv run python bench/publish_results.py [--split val] [--data-dir DIR] [--results DIR]

Reads <data dir>/bench/<source> (see FILES; missing ones are skipped and listed), writes
results/benchmark/<split>/<stable name>. Absolute paths under the data dir are rewritten relative
to it; any other absolute path aborts. The CPU-time line of each .md (evaluate.py keeps timings
out of its JSON) is added to the JSON as "cpu_s_per_plate". comparison.md is compare.py's tables
over the published JSONs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from compare import compare
from dotenv import load_dotenv

from colonycount._env import env_path

# stable name in results/ -> source under <data dir>/bench, {split} filled in
FILES = {
    "rfdetr-nano": "run1/{split}",
    "rfdetr-nano-noplate": "run1/{split}-noplate",
    "yolov8n": "yolov8n/{split}",
    "yolov8n-noplate": "yolov8n/{split}-noplate",
    "yolov8m": "yolov8m/{split}",
    "yolov8m-noplate": "yolov8m/{split}-noplate",
    "yolox-tiny": "yolox_tiny/{split}",
    "yolox-tiny-noplate": "yolox_tiny/{split}-noplate",
    "yolox-m": "yolox_m/{split}",
    "yolox-m-noplate": "yolox_m/{split}-noplate",
    "opencfu": "final/opencfu-{split}",
    "opencfu-noplate": "final/opencfu-{split}-noplate",
    "opencfu-grid.jsonl": "opencfu-{split}-grid-partial.jsonl",
    "opencfu-noplate-grid.jsonl": "opencfu-{split}-noplate-grid-partial.jsonl",
}
LABELS = {  # comparison.md row labels, in this order; compare.py appends (crop)/(whole image)
    "rfdetr-nano": "RF-DETR-nano",
    "yolov8n": "YOLOv8-n",
    "yolov8m": "YOLOv8-m",
    "yolox-tiny": "YOLOX-Tiny",
    "yolox-m": "YOLOX-M",
    "opencfu": "OpenCFU",
    "rfdetr-nano-noplate": "RF-DETR-nano",
    "yolov8n-noplate": "YOLOv8-n",
    "yolov8m-noplate": "YOLOv8-m",
    "yolox-tiny-noplate": "YOLOX-Tiny",
    "yolox-m-noplate": "YOLOX-M",
    "opencfu-noplate": "OpenCFU",
}
ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]|(?<![\w.])/(home|Users|root|tmp|mnt|data)/|~[\\/]")
CPU = re.compile(r"CPU time per plate: median ([\d.]+) s, mean ([\d.]+) s, max ([\d.]+) s")


def relative(obj, root: Path):
    """Every string that is a path under root becomes relative to it (posix slashes)."""
    if isinstance(obj, dict):
        return {k: relative(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relative(v, root) for v in obj]
    if isinstance(obj, str) and (Path(obj).is_absolute() or ABSOLUTE.search(obj)):
        try:
            return Path(obj).resolve().relative_to(root).as_posix()
        except ValueError:
            raise ValueError(f"absolute path outside the data dir: {obj!r}") from None
    return obj


def comparison(folder: Path) -> str:
    reps = [
        (label, json.loads((folder / f"{name}.json").read_text()))
        for name, label in LABELS.items()
        if (folder / f"{name}.json").exists()
    ]
    return compare(reps) + "\n"


def publish(data_dir: Path, results: Path, split: str) -> list[str]:
    """Copy what exists; return the stable names that were missing."""
    root, src_dir = data_dir.resolve(), data_dir / "bench"
    out = results / "benchmark" / split
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, src in FILES.items():
        src = src_dir / src.format(split=split)
        if name.endswith(".jsonl"):
            if not src.exists():
                missing.append(name)
                continue
            lines = [relative(json.loads(x), root) for x in src.read_text().splitlines() if x]
            (out / name).write_text("".join(json.dumps(x) + "\n" for x in lines))
            continue
        js, md = src.with_name(src.name + ".json"), src.with_name(src.name + ".md")
        if not (js.exists() and md.exists()):
            missing.append(name)
            continue
        rep, text = relative(json.loads(js.read_text()), root), md.read_text()
        if ABSOLUTE.search(text):
            raise ValueError(f"absolute path in {md}")
        if m := CPU.search(text):
            rep["cpu_s_per_plate"] = dict(
                zip(["median", "mean", "max"], map(float, m.groups()), strict=True)
            )
        (out / f"{name}.json").write_text(json.dumps(rep, indent=1) + "\n")
        (out / f"{name}.md").write_text(text)
    (out / "comparison.md").write_text(comparison(out))
    return missing


def main() -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data-dir", type=Path, help="default: COLONYCOUNT_DATA_DIR")
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    data_dir = args.data_dir or env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
    missing = publish(data_dir, args.results, args.split)
    print("published; missing (not produced yet):", ", ".join(missing) or "none")


if __name__ == "__main__":
    main()
