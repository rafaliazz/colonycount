"""One comparison table from several benchmark JSONs (evaluate.py / opencfu_runner.py output).

Usage: uv run python bench/compare.py "RF-DETR-nano=bench/run1/val.json" "OpenCFU=opencfu-val.json"
Prints markdown: one table per source (and overall) with a row per method, then per density bucket.
"""

import json
import sys
from pathlib import Path

COLS = ["n", "mae", "bias", "mdape_30_300", "n_30_300", "precision", "recall", "plate_found"]


def table(reps: list[tuple[str, dict]], title: str, rows) -> list[str]:
    out = [f"| {title} | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for label, rep in reps:
        crop = "crop" if rep["plate_restriction"] else "whole image"
        r = rows(rep)
        if r:
            out.append(
                f"| {label} ({crop}) | "
                + " | ".join("-" if r[c] is None else str(r[c]) for c in COLS)
                + " |"
            )
    return [*out, ""]


def summary(reps: list[tuple[str, dict]]) -> list[str]:
    """Method, setting, threshold and CPU s/plate (only if publish_results.py added it)."""
    out = ["| method | setting | threshold | CPU s/plate (median) |", "|---|---|---|---|"]
    for label, rep in reps:
        crop = "crop" if rep["plate_restriction"] else "whole image"
        cpu = rep.get("cpu_s_per_plate", {}).get("median", "-")
        setting, t = (rep["flags"], "-") if rep.get("flags") else (rep["model"], rep["threshold"])
        out.append(f"| {label} ({crop}) | {setting} | {t} | {cpu} |")
    return [*out, ""]


def compare(reps: list[tuple[str, dict]]) -> str:
    first = reps[0][1]
    out = summary(reps)
    for s in first["by_source"]:
        out += table(reps, s, lambda rep, s=s: rep["by_source"].get(s))
    for s, buckets in first["by_density"].items():
        for b in buckets:
            out += table(
                reps, f"{s}, GT {b}", lambda rep, s=s, b=b: rep["by_density"].get(s, {}).get(b)
            )
    return "\n".join(out)


if __name__ == "__main__":
    args = [a.split("=", 1) for a in sys.argv[1:]]
    print(compare([(k, json.loads(Path(v).read_text())) for k, v in args]))
