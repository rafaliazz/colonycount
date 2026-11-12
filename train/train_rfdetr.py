"""Train RF-DETR-nano on the tiles from make_tiles.py (task 2.1), sized for a 4 GB GPU.

Usage: uv run --no-sync python train/train_rfdetr.py [--data-dir DIR] [--epochs N]
       uv run --no-sync python train/train_rfdetr.py --resume <run dir>/last.ckpt [--epochs N]

Writes <data-dir>/runs/rfdetr_nano/<timestamp>/: rfdetr's metrics.csv (val/mAP_50 and
val/mAP_50_95 per epoch), checkpoint_best_total.pth (best val mAP@0.5:0.95, EMA or regular),
last.ckpt (saved after every epoch), and summary.json (config, wall time, peak VRAM).

--resume continues a stopped run from its last.ckpt in the same run directory, restoring the
epoch, optimizer and LR schedule; --epochs is the total for the whole run, not extra epochs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from colonycount.detection.tiling import TILE_SIZE

SEED = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path.home() / "data" / "colonycount")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--resume", type=Path, help="last.ckpt of a stopped run to continue")
    args = ap.parse_args(argv)
    if args.resume is not None and not args.resume.is_file():
        ap.error(f"--resume: no checkpoint at {args.resume}")
    return args


def build_config(args: argparse.Namespace) -> dict:
    if args.resume is not None:
        out = args.resume.parent  # keep one run dir, so metrics.csv stays a single log
    else:
        out = args.data_dir / "runs" / "rfdetr_nano" / time.strftime("%Y%m%d-%H%M%S")
    config = dict(
        dataset_dir=str(args.data_dir / "public" / f"tiles_{TILE_SIZE}"),
        output_dir=str(out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        resolution=TILE_SIZE,  # native-resolution tiles, as at inference
        seed=SEED,
        amp_dtype="auto",
        early_stopping=True,
        early_stopping_patience=5,
        tensorboard=False,
        progress_bar="tqdm",
    )
    if args.resume is not None:
        config["resume"] = str(args.resume)
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)

    import torch  # train extra only; kept out of module scope so CI can test the config
    from rfdetr import RFDETRNano

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    RFDETRNano().train(**config)
    summary = {
        "config": config,
        "wall_s": round(time.time() - t0),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    out = Path(config["output_dir"])
    name = "summary_resumed.json" if args.resume is not None else "summary.json"
    (out / name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
