"""Train the Ultralytics YOLOv8-n / -m baseline on the RF-DETR tiles (task 2.2), benchmark only.

Ultralytics is AGPL-3.0: it lives in the `baseline` dependency group, never in the package.
Usage: uv sync --extra train --group baseline, then
       uv run --no-sync python train/train_yolo.py --size n|m [--epochs N] [--batch B]
       uv run --no-sync python train/train_yolo.py --size n --resume <run dir>/weights/last.pt

Converts <data-dir>/public/tiles_<T>/{train,valid} (make_tiles.py, COCO) to the YOLO layout in
<data-dir>/public/yolo_<T>/ (hard-linked jpgs, labels/*.txt, data.yaml), then trains from the COCO
weights at imgsz T, single class, seed COLONYCOUNT_SEED, early stopping (patience 5). Writes
<data-dir>/runs/yolov8<size>/<timestamp>/: results.csv (val metrics/mAP50(B) and mAP50-95(B) per
epoch), weights/best.pt + last.pt, summary.json (config, wall time, peak VRAM).

--batch -1 (default) lets Ultralytics' AutoBatch pick the batch for ~60% of GPU memory. --resume
continues in the same run dir with the checkpoint's own settings (epochs included); only batch,
workers and patience can change. --fraction trains on part of train (smoke runs).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv

from colonycount._env import env_int, env_path

SPLITS = ("train", "valid")  # make_tiles.py folder names, kept for YOLO


def coco_to_yolo(tiles_dir: Path, out: Path) -> Path:
    """Write the YOLO dataset for tiles_dir to out; return its data.yaml. Re-running is cheap."""
    for split in SPLITS:
        coco = json.loads((tiles_dir / split / "_annotations.coco.json").read_text())
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        lines: dict[int, list[str]] = {i["id"]: [] for i in coco["images"]}
        size = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            iw, ih = size[a["image_id"]]
            lines[a["image_id"]].append(
                f"0 {(x + w / 2) / iw:.6f} {(y + h / 2) / ih:.6f} {w / iw:.6f} {h / ih:.6f}"
            )
        for im in coco["images"]:
            src, dst = tiles_dir / split / im["file_name"], out / split / "images" / im["file_name"]
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:  # other drive / no hard links
                    shutil.copy2(src, dst)
            label = out / split / "labels" / (Path(im["file_name"]).stem + ".txt")
            label.write_text("".join(s + "\n" for s in lines[im["id"]]))
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.as_posix()}\ntrain: train/images\nval: valid/images\nnames:\n  0: colony\n"
    )
    return data_yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["n", "m"], required=True, help="YOLOv8-n or YOLOv8-m")
    ap.add_argument("--data-dir", type=Path, help="default: COLONYCOUNT_DATA_DIR")
    ap.add_argument("--epochs", type=int, help="default: COLONYCOUNT_TRAIN_EPOCHS")
    ap.add_argument("--batch", type=float, default=-1, help="-1 = AutoBatch; 0..1 = GPU fraction")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of train tiles to use")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", type=Path, help="weights/last.pt of a stopped run to continue")
    args = ap.parse_args(argv)
    try:
        args.data_dir = args.data_dir or env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
        if args.epochs is None:
            args.epochs = env_int("COLONYCOUNT_TRAIN_EPOCHS", 20, lo=1)
    except ValueError as e:
        ap.error(str(e))
    if args.resume is not None and not args.resume.is_file():
        ap.error(f"--resume: no checkpoint at {args.resume}")
    return args


def build_config(args: argparse.Namespace, data_yaml: Path) -> dict:
    """Keyword arguments for ultralytics YOLO.train()."""
    if args.resume is not None:
        return dict(resume=str(args.resume), batch=_batch(args.batch), workers=args.workers)
    tile = env_int("COLONYCOUNT_TRAIN_TILE_SIZE", 640, lo=1)
    return dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=tile,  # native-resolution tiles, as at inference
        batch=_batch(args.batch),
        fraction=args.fraction,
        workers=args.workers,
        seed=env_int("COLONYCOUNT_SEED", 0),
        deterministic=True,
        single_cls=True,
        max_det=300,  # dense tiles; inference caps per tile pass, tiling keeps plates uncapped
        patience=5,
        project=str(args.data_dir / "runs" / f"yolov8{args.size}"),
        name=time.strftime("%Y%m%d-%H%M%S"),
        plots=False,
    )


def _batch(b: float) -> int | float:
    return int(b) if b == int(b) else b  # Ultralytics: int = batch size, float = GPU fraction


def main() -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    args = parse_args()
    tile = env_int("COLONYCOUNT_TRAIN_TILE_SIZE", 640, lo=1)
    tiles = args.data_dir / "public" / f"tiles_{tile}"
    data_yaml = coco_to_yolo(tiles, args.data_dir / "public" / f"yolo_{tile}")
    config = build_config(args, data_yaml)

    import torch  # train extra + baseline group only; CI tests the config without them
    from ultralytics import YOLO

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = YOLO(str(args.resume) if args.resume else f"yolov8{args.size}.pt")
    model.train(**config)
    out = Path(model.trainer.save_dir)
    summary = {
        "config": config,
        "batch": model.trainer.batch_size,
        "wall_s": round(time.time() - t0),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    name = "summary_resumed.json" if args.resume is not None else "summary.json"
    (out / name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
