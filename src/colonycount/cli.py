"""Count colonies in a batch of plate photos and write one CSV row per image."""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

import colonycount
from colonycount._env import env_str
from colonycount.detection.pipeline import count
from colonycount.models import registry
from colonycount.utils import cfu

COLUMNS = [
    "file",
    "count",
    "plate_found",
    "range_flag",
    "threshold",
    "model",
    "model_version",
    "package_version",
    "cfu_per_ml",
    "error",
]


def main(argv: list[str] | None = None) -> int:
    load_dotenv(".env")  # from the working directory; real environment variables win
    p = argparse.ArgumentParser(prog="colonycount", description=__doc__)
    p.add_argument("images", nargs="*", help="image files or glob patterns (e.g. plates/*.jpg)")
    p.add_argument("--dilution", type=float, help="dilution factor of the plated sample, e.g. 1e-4")
    p.add_argument("--volume", type=float, help="plated volume in mL, e.g. 0.1")
    p.add_argument(
        "--threshold",
        type=float,
        help="score threshold (COLONYCOUNT_THRESHOLD; default: the model's own)",
    )
    p.add_argument("--model", help="built-in model name, see --list-models (COLONYCOUNT_MODEL)")
    p.add_argument(
        "--model-path", help="local ONNX file instead of a built-in model (COLONYCOUNT_MODEL_PATH)"
    )
    p.add_argument(
        "--range",
        nargs=2,
        type=int,
        default=(),
        metavar=("LOW", "HIGH"),
        help="countable range for TFTC/TNTC flags (COLONYCOUNT_COUNTABLE_RANGE, default: 30 300)",
    )
    p.add_argument("-o", "--output", help="CSV file to write (default: stdout)")
    p.add_argument(
        "--detections", metavar="DIR", help="also write one JSON of detections per image"
    )
    p.add_argument("--list-models", action="store_true", help="list built-in models and exit")
    args = p.parse_args(argv)

    if args.list_models:
        for m in registry.MODELS.values():
            print(f"{m.name}\t{m.version}\t{m.license}")
        return 0
    if not args.images:
        p.error("no images given")
    want_cfu = args.dilution is not None or args.volume is not None
    try:
        args.model = args.model or env_str("COLONYCOUNT_MODEL", "public")
        registry.get(args.model)
        if args.threshold is None:
            args.threshold = registry.threshold(args.model, args.model_path)
        cfu.range_flag(0, *args.range)
        if want_cfu:
            cfu.cfu_per_ml(0, args.dilution, args.volume)
    except ValueError as e:
        p.error(str(e))

    files = []
    for pattern in args.images:  # Windows shells don't expand globs
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        files += matches or [pattern]  # an unmatched pattern becomes an error row
    if args.detections:
        Path(args.detections).mkdir(parents=True, exist_ok=True)

    out = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    writer = csv.DictWriter(out, COLUMNS)
    writer.writeheader()
    failed = False
    for f in files:
        row = {"file": f, "threshold": args.threshold, "model": args.model}
        row["package_version"] = colonycount.__version__
        try:
            r = count(f, threshold=args.threshold, model=args.model, model_path=args.model_path)
            row |= {
                "count": r.total,
                "plate_found": r.plate_found,
                "range_flag": cfu.range_flag(r.total, *args.range),
                "model_version": r.model_version,
            }
            if want_cfu:
                row["cfu_per_ml"] = cfu.cfu_per_ml(r.total, args.dilution, args.volume)
            if args.detections:
                # ponytail: named by file stem, so same-named images in different folders overwrite
                (Path(args.detections) / f"{Path(f).stem}.json").write_text(
                    json.dumps(
                        {
                            "file": f,
                            "model": r.model,
                            "model_version": r.model_version,
                            "threshold": r.threshold,
                            "plate_found": r.plate_found,
                            "detections": [vars(d) for d in r.detections],
                        },
                        indent=1,
                    )
                )
        except Exception as e:  # one bad image must not stop the batch
            row["error"] = f"{type(e).__name__}: {e}"
            failed = True
        writer.writerow(row)
    if out is not sys.stdout:
        out.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
