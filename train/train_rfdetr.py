"""Train RF-DETR-nano on the tiles from make_tiles.py (task 2.1), sized for a 4 GB GPU.

Usage: uv run --extra train python train/train_rfdetr.py [--data-dir DIR] [--epochs N]

Writes <data-dir>/runs/rfdetr_nano/<timestamp>/: rfdetr's metrics.csv (val/mAP_50 and
val/mAP_50_95 per epoch), checkpoint_best_total.pth (best val mAP@0.5:0.95, EMA or regular),
last.ckpt, and summary.json (config, wall time, peak VRAM).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from rfdetr import RFDETRNano

from colonycount.detection.tiling import TILE_SIZE

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path.home() / "data" / "colonycount")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

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
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    RFDETRNano().train(**config)
    summary = {
        "config": config,
        "wall_s": round(time.time() - t0),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
