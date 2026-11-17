"""Copy benchmark outputs from the data folder into the repo's results/ and rebuild comparison.md.

Usage: uv run python bench/publish_results.py [--split val] [--data-dir DIR] [--results DIR]

Reads <data dir>/bench/<source> (see FILES; missing ones are skipped and listed), writes
results/benchmark/<split>/<stable name>. Absolute paths under the data dir are rewritten relative
to it; any other absolute path aborts. The CPU-time line of each .md (evaluate.py keeps timings
out of its JSON) is added to the JSON as "cpu_s_per_plate". comparison.md is compare.py's tables
over the published JSONs. Then the generated blocks of results/README.md and README.md (between
<!-- results:begin --> and <!-- results:end -->) are rebuilt from those JSONs; --reports-only does
just that, without needing the data folder.
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
    "rfdetr-nano": "rfdetr-nano/{split}",
    "rfdetr-nano-noplate": "rfdetr-nano/{split}-noplate",
    "yolov8n": "yolov8n/{split}",
    "yolov8n-noplate": "yolov8n/{split}-noplate",
    "yolov8m": "yolov8m/{split}",
    "yolov8m-noplate": "yolov8m/{split}-noplate",
    "yolox-tiny": "yolox_tiny/{split}",
    "yolox-tiny-noplate": "yolox_tiny/{split}-noplate",
    "yolox-m": "yolox_m/{split}",
    "yolox-m-noplate": "yolox_m/{split}-noplate",
    "opencfu": "opencfu/opencfu-{split}",
    "opencfu-noplate": "opencfu/opencfu-{split}-noplate",
    "opencfu-grid.jsonl": "opencfu/opencfu-{split}-grid.json",
    "opencfu-noplate-grid.jsonl": "opencfu/opencfu-{split}-noplate-grid.json",
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
# base name -> (shipped in v0.1, license, parameters); the JSONs hold neither (parameters:
# counted from each exported ONNX file's weight tensors, 2026-11-15)
INFO = {
    "rfdetr-nano": (True, "Apache-2.0", "27.2 M"),
    "yolox-tiny": (True, "Apache-2.0", "5.0 M"),
    "yolox-m": (True, "Apache-2.0", "25.3 M"),
    "yolov8n": (False, "AGPL-3.0", "3.1 M"),
    "yolov8m": (False, "AGPL-3.0", "25.8 M"),
    "opencfu": (False, "GPL-3.0 (tool)", "-"),
}
MODES = [("", "crop"), ("-noplate", "whole image")]  # first = the library default (crop)
BLOCK = re.compile(r"<!-- results:begin -->.*<!-- results:end -->", re.S)
SHORT_HEAD = ["model", "MdAPE (30-300)", "MAE", "CPU s/plate", "license"]
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


def _table(head, rows) -> list[str]:
    row = lambda cells: "| " + " | ".join(map(str, cells)) + " |"  # noqa: E731
    return [row(head), "|---" * len(head) + "|", *map(row, rows), ""]


def _load(folder: Path) -> dict:
    return {
        n: json.loads((folder / f"{n}.json").read_text())
        for n in LABELS
        if (folder / f"{n}.json").exists()
    }


def _name(base: str) -> str:
    return f"**{LABELS[base]}** (ships)" if INFO[base][0] else LABELS[base]


def _headline_rows(reps: dict, suffix: str, short: bool):
    for base, (shipped, lic, params) in INFO.items():
        if not (r := reps.get(base + suffix)) or (short and not (shipped or base == "opencfu")):
            continue
        o, cpu = r["by_source"]["overall"], r["cpu_s_per_plate"]["median"]
        row = [_name(base), f"{o['mdape_30_300']:.2f}%", f"{o['mae']:.1f}"]
        if short:
            yield [*row, f"{cpu:.1f}", lic]
        else:
            t = "-" if r.get("flags") else r["threshold"]
            yield [
                *row,
                f"{o['bias']:+.1f}",
                f"{o['precision']:.1%}",
                f"{o['recall']:.1%}",
                f"{cpu:.1f}",
                t,
                lic,
                params,
            ]


def readme_summary(folder: Path, link: str) -> str:
    """The short block of the root README: shipped models + OpenCFU, default (crop) mode."""
    reps = _load(folder)
    split = reps["rfdetr-nano"]["split"]
    return "\n".join(
        [
            "<!-- results:begin -->",
            f"On the {split} split, plate crop on (the default). MdAPE is the median error of the "
            "plate count on plates with 30-300 colonies, MAE the mean error in colonies per "
            "plate, and CPU time is the median per plate:",
            "",
            *_table(SHORT_HEAD, _headline_rows(reps, "", True)),
            (
                f"These are {split}-split numbers, slightly optimistic because the thresholds were "
                "chosen on it."
                if split == "val"
                else "Held-out test split (unseen plates), run once with every setting fixed on "
                "the validation split."
            )
            + f" Full report: [{link}]({link}).",
            "<!-- results:end -->",
        ]
    )


def report(folder: Path) -> str:
    """The generated block of results/README.md, from the JSONs in folder."""
    reps = _load(folder)
    ref = reps["rfdetr-nano"]
    split, n = ref["split"], {k: v["n"] for k, v in ref["by_source"].items()}
    ship = [b for b, i in INFO.items() if i[0]]

    def mdape(k):
        return reps[k]["by_source"]["overall"]["mdape_30_300"]

    def cpu(k):
        return reps[k]["cpu_s_per_plate"]["median"]

    best, fast = min(ship, key=mdape), min(ship, key=cpu)
    whole_wins = all(mdape(b + "-noplate") < mdape(b) for b in ship)
    out = [
        "<!-- results:begin -->",
        "## Summary",
        "",
        f"This is the **{split}** split: {n['overall']} plates ({n['clinical']} clinical, "
        f"{n['smartphone']} smartphone photos). Each method counted the colonies on every plate "
        "and was compared with the human count. "
        f"The most accurate shipped model is **{LABELS[best]}** (median error "
        f"{mdape(best):.2f}% on plates with 30-300 colonies); the fastest is "
        f"**{LABELS[fast]}** ({cpu(fast):.1f} s per plate on CPU). Every shipped model is "
        f"far more accurate than the classic baseline, OpenCFU ({mdape('opencfu'):.2f}%).",
        "",
        "Models marked **(ships)** are the three built into colonycount v0.1. The YOLOv8 models "
        "are benchmark-only (AGPL-3.0) and OpenCFU is the classic non-deep-learning baseline.",
        "",
        (
            f"> These are {split}-split numbers. The detection thresholds and the best training "
            "epochs were chosen on this split, so they are slightly optimistic; the held-out "
            "test split is run once on all models at release and this report is then regenerated."
            if split == "val"
            else "> Held-out test split, run once; thresholds were fixed on the validation split."
        ),
        "",
    ]
    for suffix, mode in MODES:
        out += [
            f"## Headline: {mode}" + ("" if suffix else " (default)"),
            "",
            "All plates. MdAPE is on plates with 30-300 true colonies; bias is predicted minus "
            "true.",
            "",
            *_table(
                [*SHORT_HEAD[:3], "bias", "precision", "recall", "CPU s/plate", "threshold"]
                + ["license", "params"],
                _headline_rows(reps, suffix, False),
            ),
        ]
        if not suffix:
            out += [
                "Counting defaults to the plate crop: `colonycount.count` finds the plate rim "
                "and counts inside it. "
                + (
                    "On this split counting the whole image was more accurate for every shipped "
                    "model (next table), but the crop is listed first because it is the "
                    "default."
                    if whole_wins
                    else ""
                ),
                "",
            ]
    rows = []
    for suffix, mode in MODES:
        for base in INFO:
            if r := reps.get(base + suffix):
                c, s = r["by_source"]["clinical"], r["by_source"]["smartphone"]
                rows.append(
                    [_name(base), mode]
                    + [f"{c['mdape_30_300']:.2f}%", f"{c['mae']:.1f}"]
                    + [f"{s['mdape_30_300']:.2f}%", f"{s['mae']:.1f}"]
                )
    out += [
        "## By source",
        "",
        "Clinical plates vs smartphone photos.",
        "",
        *_table(
            [
                "model",
                "mode",
                "clinical MdAPE",
                "clinical MAE",
                "smartphone MdAPE",
                "smartphone MAE",
            ],
            rows,
        ),
    ]
    dens = ref["by_density"]["all"]
    rows = []
    for suffix, mode in MODES:
        for base in INFO:
            if r := reps.get(base + suffix):
                cells = [_name(base), mode]
                for b in dens:
                    d = r["by_density"]["all"][b]
                    empty = d["mae"] is None  # e.g. no >300 plates in a split
                    cells += ["-", "-"] if empty else [f"{d['mae']:.1f}", f"{d['mdape_all']:.2f}%"]
                rows.append(cells)
    out += [
        "## By colony density",
        "",
        "True colony count per plate: "
        + ", ".join(f"{b} ({d['n']} plates)" for b, d in dens.items())
        + ". Plates under 30 colonies are TFTC (too few to count) and plates over 300 are TNTC "
        "(too numerous to count); lab practice counts only 30-300, which is why the headline "
        "MdAPE uses that range. Here MdAPE covers all plates in the bucket.",
        "",
        *_table(["model", "mode"] + [f"{b} {m}" for b in dens for m in ("MAE", "MdAPE")], rows),
        "## CPU speed",
        "",
        "Median seconds per plate on CPU, one plate at a time (OpenCFU runs in a container on "
        "images resized to 1000 px).",
        "",
        *_table(
            ["model", *(m for _, m in MODES)],
            [
                [_name(b), *(f"{reps[b + s]['cpu_s_per_plate']['median']:.1f}" for s, _ in MODES)]
                for b in INFO
                if all(b + s in reps for s, _ in MODES)
            ],
        ),
        "## How the threshold was chosen",
        "",
        "A detection counts when its score is at least the threshold. For each model the "
        "threshold was swept automatically on the validation split from 0.20 to 0.80 and the "
        "value with the lowest MdAPE (30-300) was kept, ties broken by the lowest MAE. "
        + (
            "MdAPE by threshold (the chosen one in bold):"
            if split == "val"
            else "Below, MdAPE on this test split by threshold, for information only: the bold "
            "value is the one fixed on validation, not the best one here."
        ),
        "",
    ]
    cols = [(b + s, f"{LABELS[b]} {m}") for b in ship for s, m in MODES]
    rows = []
    for t in ref["threshold_curve"]:
        cells = [t]
        for k, _ in cols:
            v = f"{reps[k]['threshold_curve'][t]['mdape_30_300']:.2f}%"
            cells.append(f"**{v}**" if float(t) == reps[k]["threshold"] else v)
        rows.append(cells)
    out += [
        *_table(["threshold", *(c for _, c in cols)], rows),
        "## How to read these metrics",
        "",
        "- **MdAPE (30-300)**: median absolute percentage error of the plate count, on plates "
        "with 30-300 true colonies. Half of the plates are off by less than this.",
        "- **MAE**: mean absolute error, in colonies per plate (all plates).",
        "- **bias**: mean of predicted minus true count; positive means over-counting.",
        "- **precision**: share of detections that are real colonies. **recall**: share of real "
        "colonies that were found.",
        "- **match radius**: a detection matches a true colony when its centre is within half "
        "the colony's mean box side of the true centre; each colony matches at most once.",
        "<!-- results:end -->",
    ]
    return "\n".join(out)


def splice(text: str, block: str) -> str:
    """text with its generated block (between the results markers) replaced by block."""
    if not BLOCK.search(text):
        raise ValueError("no <!-- results:begin --> ... <!-- results:end --> block")
    return BLOCK.sub(lambda _: block, text)


def write_reports(results: Path, split: str) -> None:
    folder = results / "benchmark" / split
    for path, block in [
        (results / "README.md", report(folder)),
        (Path("README.md"), readme_summary(folder, "results/README.md")),
    ]:
        path.write_text(splice(path.read_text(), block))


def publish(data_dir: Path, results: Path, split: str) -> list[str]:
    """Copy what exists; return the stable names that were missing."""
    root, src_dir = data_dir.resolve(), data_dir / "bench"
    out = results / "benchmark" / split
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, src in FILES.items():
        src = src_dir / src.format(split=split)
        if name.endswith(".jsonl"):
            if split == "test":  # settings are tuned on val only; test has no grid
                continue
            if not src.exists():
                missing.append(name)
                continue
            rows = json.loads(src.read_text())  # opencfu_runner writes the grid as one JSON list
            lines = [relative(x, root) for x in rows]
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
    ap.add_argument("--reports-only", action="store_true", help="only rebuild the two READMEs")
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    if not args.reports_only:
        data_dir = args.data_dir or env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
        missing = publish(data_dir, args.results, args.split)
        print("published; missing (not produced yet):", ", ".join(missing) or "none")
    write_reports(args.results, args.split)


if __name__ == "__main__":
    main()
