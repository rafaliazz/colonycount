# Benchmark results

Plate-count benchmark outputs, committed so they can be read without the data. Not part of the
package (only `src/` goes into the wheel). The report below is
generated from the benchmark JSON; the full per-method tables are in
[`benchmark/val/comparison.md`](benchmark/val/comparison.md).

<!-- results:begin -->
## Summary

This is the **val** split: 198 plates (142 clinical, 56 smartphone photos). Each method counted the colonies on every plate and was compared with the human count. The most accurate shipped model is **RF-DETR-nano** (median error 2.20% on plates with 30-300 colonies); the fastest is **YOLOX-Tiny** (2.4 s per plate on CPU). Every shipped model is far more accurate than the classic baseline, OpenCFU (9.99%).

Models marked **(ships)** are the three built into colonycount v0.1. The YOLOv8 models are benchmark-only (AGPL-3.0) and OpenCFU is the classic non-deep-learning baseline.

> These are val-split numbers. The detection thresholds and the best training epochs were chosen on this split, so they are slightly optimistic; the held-out test split is run once on all models at release and this report is then regenerated.

## Headline: crop (default)

All plates. MdAPE is on plates with 30-300 true colonies; bias is predicted minus true.

| model | MdAPE (30-300) | MAE | bias | precision | recall | CPU s/plate | threshold | license | params |
|---|---|---|---|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | 2.20% | 5.8 | -1.2 | 96.6% | 95.4% | 14.3 | 0.45 | Apache-2.0 | 30 M |
| **YOLOX-Tiny** (ships) | 2.72% | 6.2 | -1.5 | 96.3% | 94.8% | 2.4 | 0.5 | Apache-2.0 | 5 M |
| **YOLOX-M** (ships) | 2.64% | 7.2 | -2.5 | 96.3% | 93.8% | 4.3 | 0.45 | Apache-2.0 | 25 M |
| YOLOv8-n | 2.38% | 7.8 | +0.4 | 94.5% | 95.0% | 1.0 | 0.3 | AGPL-3.0 | 3 M |
| YOLOv8-m | 2.23% | 7.9 | -0.7 | 95.1% | 94.4% | 9.5 | 0.3 | AGPL-3.0 | 26 M |
| OpenCFU | 9.99% | 23.2 | -9.5 | 82.5% | 74.3% | 0.6 | - | GPL-3.0 (tool) | - |

Counting defaults to the plate crop: `colonycount.count` finds the plate rim and counts inside it. On this split counting the whole image was more accurate for every shipped model (next table), but the crop is listed first because it is the default.

## Headline: whole image

All plates. MdAPE is on plates with 30-300 true colonies; bias is predicted minus true.

| model | MdAPE (30-300) | MAE | bias | precision | recall | CPU s/plate | threshold | license | params |
|---|---|---|---|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | 1.43% | 5.0 | +0.6 | 96.2% | 96.8% | 24.5 | 0.45 | Apache-2.0 | 30 M |
| **YOLOX-Tiny** (ships) | 1.43% | 5.6 | +0.2 | 95.9% | 96.1% | 4.0 | 0.5 | Apache-2.0 | 5 M |
| **YOLOX-M** (ships) | 1.56% | 6.6 | -0.8 | 95.9% | 95.1% | 7.2 | 0.45 | Apache-2.0 | 25 M |
| YOLOv8-n | 1.56% | 8.6 | +0.9 | 94.1% | 95.0% | 1.8 | 0.35 | AGPL-3.0 | 3 M |
| YOLOv8-m | 1.44% | 9.2 | -0.8 | 94.6% | 93.8% | 8.2 | 0.35 | AGPL-3.0 | 26 M |
| OpenCFU | 8.66% | 22.7 | -9.8 | 84.7% | 76.0% | 0.5 | - | GPL-3.0 (tool) | - |

## By source

Clinical plates vs smartphone photos.

| model | mode | clinical MdAPE | clinical MAE | smartphone MdAPE | smartphone MAE |
|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | crop | 2.54% | 3.6 | 1.70% | 11.2 |
| **YOLOX-Tiny** (ships) | crop | 2.84% | 3.7 | 2.05% | 12.5 |
| **YOLOX-M** (ships) | crop | 2.79% | 3.6 | 2.04% | 16.2 |
| YOLOv8-n | crop | 2.34% | 4.8 | 3.24% | 15.4 |
| YOLOv8-m | crop | 2.31% | 4.8 | 2.11% | 15.9 |
| OpenCFU | crop | 9.62% | 12.6 | 11.07% | 50.1 |
| **RF-DETR-nano** (ships) | whole image | 1.65% | 3.2 | 1.26% | 9.6 |
| **YOLOX-Tiny** (ships) | whole image | 1.39% | 3.3 | 1.73% | 11.2 |
| **YOLOX-M** (ships) | whole image | 1.67% | 3.1 | 1.44% | 15.4 |
| YOLOv8-n | whole image | 1.82% | 5.2 | 1.40% | 17.0 |
| YOLOv8-m | whole image | 1.42% | 5.0 | 1.45% | 19.8 |
| OpenCFU | whole image | 8.82% | 12.6 | 7.63% | 48.5 |

## By colony density

True colony count per plate: <30 (46 plates), 30-300 (140 plates), >300 (12 plates). Plates under 30 colonies are TFTC (too few to count) and plates over 300 are TNTC (too numerous to count); lab practice counts only 30-300, which is why the headline MdAPE uses that range. Here MdAPE covers all plates in the bucket.

| model | mode | <30 MAE | <30 MdAPE | 30-300 MAE | 30-300 MdAPE | >300 MAE | >300 MdAPE |
|---|---|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | crop | 2.1 | 10.32% | 5.0 | 2.20% | 28.6 | 2.90% |
| **YOLOX-Tiny** (ships) | crop | 2.1 | 7.85% | 5.5 | 2.72% | 29.3 | 2.86% |
| **YOLOX-M** (ships) | crop | 2.2 | 6.62% | 6.0 | 2.64% | 40.8 | 3.71% |
| YOLOv8-n | crop | 3.7 | 14.29% | 6.5 | 2.38% | 38.3 | 5.15% |
| YOLOv8-m | crop | 3.8 | 16.67% | 6.4 | 2.23% | 41.2 | 5.25% |
| OpenCFU | crop | 9.9 | 47.50% | 17.9 | 9.99% | 136.3 | 18.85% |
| **RF-DETR-nano** (ships) | whole image | 2.4 | 14.71% | 4.2 | 1.43% | 24.4 | 2.56% |
| **YOLOX-Tiny** (ships) | whole image | 2.6 | 11.11% | 4.5 | 1.43% | 28.8 | 2.40% |
| **YOLOX-M** (ships) | whole image | 2.6 | 11.11% | 5.0 | 1.56% | 40.1 | 4.02% |
| YOLOv8-n | whole image | 5.2 | 33.33% | 6.7 | 1.56% | 43.6 | 4.67% |
| YOLOv8-m | whole image | 5.2 | 34.17% | 7.0 | 1.44% | 50.3 | 4.26% |
| OpenCFU | whole image | 10.1 | 22.84% | 16.9 | 8.66% | 139.6 | 21.60% |

## CPU speed

Median seconds per plate on CPU, one plate at a time (OpenCFU runs in a container on images resized to 1000 px).

| model | crop | whole image |
|---|---|---|
| **RF-DETR-nano** (ships) | 14.3 | 24.5 |
| **YOLOX-Tiny** (ships) | 2.4 | 4.0 |
| **YOLOX-M** (ships) | 4.3 | 7.2 |
| YOLOv8-n | 1.0 | 1.8 |
| YOLOv8-m | 9.5 | 8.2 |
| OpenCFU | 0.6 | 0.5 |

## How the threshold was chosen

A detection counts when its score is at least the threshold. For each model the threshold was swept automatically on the validation split from 0.20 to 0.80 and the value with the lowest MdAPE (30-300) was kept, ties broken by the lowest MAE. MdAPE by threshold (the chosen one in bold):

| threshold | RF-DETR-nano crop | RF-DETR-nano whole image | YOLOX-Tiny crop | YOLOX-Tiny whole image | YOLOX-M crop | YOLOX-M whole image |
|---|---|---|---|---|---|---|
| 0.2 | 3.13% | 2.76% | 3.00% | 2.78% | 2.87% | 2.18% |
| 0.25 | 2.88% | 2.52% | 2.81% | 2.65% | 2.78% | 2.10% |
| 0.3 | 2.63% | 1.90% | 2.86% | 2.51% | 2.74% | 2.02% |
| 0.35 | 2.30% | 1.67% | 2.84% | 2.33% | 2.69% | 1.67% |
| 0.4 | 2.37% | 1.43% | 2.86% | 2.13% | 2.75% | 1.67% |
| 0.45 | **2.20%** | **1.43%** | 2.84% | 1.82% | **2.64%** | **1.56%** |
| 0.5 | 2.27% | 1.49% | **2.72%** | **1.43%** | 2.75% | 1.60% |
| 0.55 | 2.72% | 1.67% | 2.72% | 1.45% | 2.78% | 1.61% |
| 0.6 | 3.19% | 1.99% | 2.76% | 1.61% | 2.99% | 2.06% |
| 0.65 | 4.27% | 2.20% | 3.29% | 2.27% | 4.22% | 2.67% |
| 0.7 | 5.08% | 3.07% | 5.24% | 2.79% | 5.75% | 2.95% |
| 0.75 | 6.59% | 4.20% | 6.72% | 3.40% | 6.42% | 3.82% |
| 0.8 | 8.81% | 5.58% | 8.30% | 5.04% | 7.50% | 4.35% |

## How to read these metrics

- **MdAPE (30-300)**: median absolute percentage error of the plate count, on plates with 30-300 true colonies. Half of the plates are off by less than this.
- **MAE**: mean absolute error, in colonies per plate (all plates).
- **bias**: mean of predicted minus true count; positive means over-counting.
- **precision**: share of detections that are real colonies. **recall**: share of real colonies that were found.
- **match radius**: a detection matches a true colony when its centre is within half the colony's mean box side of the true centre; each colony matches at most once.
<!-- results:end -->

## Files (`benchmark/val/`)

| file | what |
|---|---|
| `rfdetr-nano.json` / `.md` | colonycount with the RF-DETR-nano model, plate crop on, threshold 0.45 (`bench/evaluate.py`) |
| `rfdetr-nano-noplate.json` / `.md` | same, whole image (`--no-plate`) |
| `yolov8n.json` / `.md` | colonycount with the YOLOv8-n model, plate crop on, threshold 0.3 (`bench/evaluate.py`) |
| `yolov8n-noplate.json` / `.md` | same, whole image, threshold 0.35 |
| `yolov8m.json` / `.md` | colonycount with the YOLOv8-m model, plate crop on, threshold 0.3 (`bench/evaluate.py`) |
| `yolov8m-noplate.json` / `.md` | same, whole image, threshold 0.35 |
| `yolox-tiny.json` / `.md` | colonycount with the YOLOX-Tiny model, plate crop on, threshold 0.5 (`bench/evaluate.py`) |
| `yolox-tiny-noplate.json` / `.md` | same, whole image, threshold 0.5 |
| `yolox-m.json` / `.md` | colonycount with the YOLOX-M model, plate crop on, threshold 0.45 (`bench/evaluate.py`) |
| `yolox-m-noplate.json` / `.md` | same, whole image, threshold 0.45 |
| `opencfu.json` / `.md` | OpenCFU baseline, plate crop on, best tuned setting (`bench/opencfu_runner.py`) |
| `opencfu-noplate.json` / `.md` | same, whole image |
| `opencfu-grid.jsonl` | OpenCFU settings grid on val, plate crop: one JSON line per setting (partial, see caveats) |
| `opencfu-noplate-grid.jsonl` | same, whole image (partial) |
| `comparison.md` | all methods side by side, per source, per density bucket and overall (`bench/compare.py`) |

Columns: `mae` and `bias` (predicted minus true) in colonies per plate; `mdape_30_300` is the
median absolute percentage error on plates whose true count is 30-300 (`n_30_300` of them);
`precision`/`recall` are per colony; `plate_found` is the share of plates where the plate rim was
found (0 with the crop off). CPU s/plate is the median wall time per plate on CPU, one plate at a
time: around `colonycount.count` for the model, around the `opencfu` call inside the container for
OpenCFU (which sees images resized to a 1000 px longest side).

The JSONs are the tools' outputs with two changes made by `bench/publish_results.py`: paths are
relative to `COLONYCOUNT_DATA_DIR` (none are absolute), and `cpu_s_per_plate` (median/mean/max,
from the `.md`'s CPU-time line, since the JSONs are kept free of timings) is added.

## Commands

Equivalent to the runs above (the output folders match the data layout that
`publish_results.py` reads):

```powershell
# RF-DETR-nano (writes $env:COLONYCOUNT_DATA_DIR/bench/run1/val[-noplate].json/.md)
uv run python bench/evaluate.py --model "$env:COLONYCOUNT_DATA_DIR/models/public-rfdetr-nano-20261113.onnx" --split val --out "$env:COLONYCOUNT_DATA_DIR/bench/run1"
uv run python bench/evaluate.py --model "$env:COLONYCOUNT_DATA_DIR/models/public-rfdetr-nano-20261113.onnx" --split val --no-plate --out "$env:COLONYCOUNT_DATA_DIR/bench/run1"

# YOLOv8-n / YOLOv8-m (writes bench/yolov8n|yolov8m/val[-noplate].json/.md); same for yolov8m
uv run python bench/evaluate.py --model "$env:COLONYCOUNT_DATA_DIR/models/yolov8n-20261113.onnx" --split val --out "$env:COLONYCOUNT_DATA_DIR/bench/yolov8n"
uv run python bench/evaluate.py --model "$env:COLONYCOUNT_DATA_DIR/models/yolov8n-20261113.onnx" --split val --no-plate --out "$env:COLONYCOUNT_DATA_DIR/bench/yolov8n"

# YOLOX-Tiny / YOLOX-M (writes bench/yolox_tiny|yolox_m/val[-noplate].json/.md); same for yolox_m
uv run python bench/evaluate.py --model "$env:COLONYCOUNT_DATA_DIR/models/yolox_tiny-20261114.onnx" --split val --out "$env:COLONYCOUNT_DATA_DIR/bench/yolox_tiny"
uv run python bench/evaluate.py --model "$env:COLONYCOUNT_DATA_DIR/models/yolox_tiny-20261114.onnx" --split val --no-plate --out "$env:COLONYCOUNT_DATA_DIR/bench/yolox_tiny"

# OpenCFU: settings grid (its printed lines are the -grid-partial.jsonl files), then the finals
docker build -t colonycount-opencfu bench/opencfu
uv run python bench/opencfu_runner.py --tune > "$env:COLONYCOUNT_DATA_DIR/bench/opencfu-val-grid-partial.jsonl"
uv run python bench/opencfu_runner.py --tune --no-plate > "$env:COLONYCOUNT_DATA_DIR/bench/opencfu-val-noplate-grid-partial.jsonl"
uv run python bench/opencfu_runner.py --setting '{"side": 1000, "mode": "bil -t 5", "r_min": 0.004, "r_max": 0.06}' --out "$env:COLONYCOUNT_DATA_DIR/bench/final"
uv run python bench/opencfu_runner.py --no-plate --setting '{"side": 1000, "mode": "bil -t 3", "r_min": 0.004, "r_max": 0.03}' --out "$env:COLONYCOUNT_DATA_DIR/bench/final"

# copy into results/ and regenerate comparison.md
uv run python bench/publish_results.py --split val
```

(POSIX shells: `$COLONYCOUNT_DATA_DIR` instead of `$env:COLONYCOUNT_DATA_DIR`; the default data
dir is `~/data/colonycount`.) The source locations `publish_results.py` reads are listed in its
`FILES` table; any it cannot find are skipped and printed.

## Provenance

- Date: 2026-11-13. Split: **val**, 198 plates (142 clinical, 56 smartphone). The test split was
  not touched; it is reserved for one final run on all models.
- Benchmark code: `bench/evaluate.py` as in commit 5c0e0bd (RF-DETR runs);
  `bench/opencfu_runner.py` as in commit 507a7b8 (OpenCFU runs).
- Model: `models/public-rfdetr-nano-20261113.onnx`, sha256
  `42e7602cbcbb4b18d1851464b660a42d744d0b86edf11c02eb62046141081586`, exported from training run
  `rfdetr_nano/20261113-010505` (best epoch 10, val mAP@0.5:0.95 0.680).
- YOLOv8 baselines (benchmark-only, not shipped: trained with Ultralytics 8.4.211, AGPL-3.0;
  `bench/evaluate.py` runs their ONNX exports through `colonycount.count` like RF-DETR):
  - YOLOv8-n: `models/yolov8n-20261113.onnx`, sha256
    `21772870a28f1f038a6ec3d2f80dcc33698bd4afaa6f10463e184e963a5c8ccd`, training run
    `yolov8n/20261113-165314` (best val tile mAP@0.5 0.965, mAP@0.5:0.95 0.685; 1.0 h).
  - YOLOv8-m: `models/yolov8m-20261113.onnx`, sha256
    `fbb8ed15dd79a88dddec8f2b80294bbc38c45f09abf398923701b8d22acc9dde`, training run
    `yolov8m/20261113-182332` (best val tile mAP@0.5 0.965, mAP@0.5:0.95 0.689; 3.4 h).
  - Shared config (the runs' `summary.json` "comparable" block): 12 epochs, imgsz 640, seed 0,
    patience 5, nbs 16, optimizer MuSGD, AMP. Per-step batch from AutoBatch was 7 (n) / 3 (m);
    Ultralytics sets accumulate = round(16 / per-step batch) = 2 / 5, so the real effective batch
    was 14 (n) / 15 (m), not 16.
  - Thresholds (crop 0.3, whole image 0.35 for both) were picked on val by the same rule as
    RF-DETR's.
- YOLOX models (date 2026-11-14; `bench/evaluate.py` runs their ONNX exports through
  `colonycount.count` like RF-DETR). Code: Megvii-BaseDetection/YOLOX vendored in
  `train/third_party/yolox` at commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee` (Apache-2.0),
  fine-tuned from the COCO-pretrained `yolox_tiny.pth` / `yolox_m.pth` of its 0.1.1rc0 release.
  - YOLOX-Tiny: `models/yolox_tiny-20261114.onnx`, sha256
    `cbc98f010e242726c781b0a4c6d6ea5a6d6c2b81ed7f28b2b30603ccacf8900e`, training run
    `yolox_tiny/20261114-120148` (best val tile mAP@0.5 0.948, mAP@0.5:0.95 0.667; 90 min,
    peak VRAM 1.31 GB).
  - YOLOX-M: `models/yolox_m-20261114.onnx`, sha256
    `392a148324c462b36777d353ded9be3db17a67f9adbecab376099183ff4500f6`, training run
    `yolox_m/20261114-135716` (best val tile mAP@0.5 0.950, mAP@0.5:0.95 0.666; 196 min,
    peak VRAM 1.87 GB).
  - Tile mAPs are pycocotools with maxDets 300.
  - Shared config (the runs' `summary.json` "comparable" block): 12 epochs, imgsz 640, seed 0,
    patience 5, nbs 16, optimizer SGD, AMP, EMA 0.9998. Per-step batch 8 (tiny) / 4 (m) with
    accumulate 2 / 4, so the effective batch is exactly 16 for both. Augmentation is YOLOX's
    default fine-tuning recipe: flip_prob 0.5, hsv_prob 1.0, HSV gains [5, 30, 30], no mosaic
    (mosaic_epochs 0, no_aug_epochs 15), L1 loss on.
  - ONNX parity vs YOLOX's own `postprocess()` (8 val tiles, score >= 0.3): Tiny 159 boxes,
    0 count mismatches, max diff 4.8e-07 score / 6.1e-05 px box; M 162 boxes, 0 count
    mismatches, max diff 8.6e-07 score / 7.6e-05 px box.
  - Thresholds (Tiny 0.5, M 0.45, same for crop and whole image) were picked on val by the same
    rule as RF-DETR's.
- v0.1 candidates (owner decision): YOLOX-Tiny (`public-fast`) and RF-DETR-nano (`public`).
  YOLOX-M is benchmark-only.
- OpenCFU 4.0.0 (Debian package `opencfu 4.0.0+dfsg-2+b2`), image from `bench/opencfu/Dockerfile`.
- Threshold 0.45 was chosen **on val** (lowest MdAPE 30-300, tie-break MAE), and OpenCFU's settings
  were also tuned on val, so all numbers here are optimistic; an unbiased number needs the test
  split.
- Matching rule (`bench/evaluate.py`): a prediction matches a true colony when its centre lies
  within half the colony box's mean side of the box centre; greedy one-to-one, nearest pairs first.
- Determinism: the RF-DETR crop run was repeated (`run2`) and `val.json`/`val.md` were
  byte-identical to `run1`.

## Caveats

- CPU s/plate is not strictly comparable between runs (runs were on different days, some from different shells): the YOLOv8-m whole-image run was done
  from the owner's terminal, while the YOLOv8-m crop run was done from a Claude Code background
  shell under memory pressure, which is part of why its CPU time per plate (9.48 s) is higher
  than the whole-image run's (8.23 s).
- The OpenCFU grid (2 sides x 7 threshold modes x 2 x 2 radius ranges) was stopped early. Plate
  crop: 27 of the 28 settings at 1000 px (`-a -r 4 -R 60` missing), none at 1500 px. Whole image:
  all 28 at 1000 px, 2 of 28 at 1500 px. The final OpenCFU runs use the best 1000 px setting of
  each grid (crop `-d bil -t 5 -r 4 -R 60`, whole image `-d bil -t 3 -r 4 -R 30`).
