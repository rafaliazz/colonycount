# Benchmark results

Plate-count benchmark outputs, committed so they can be read without the data. Not part of the
package (only `src/` goes into the wheel). The report below is
generated from the benchmark JSON; the full per-method tables are in
[`benchmark/val/comparison.md`](benchmark/val/comparison.md).

<!-- results:begin -->
## Summary

This is the **val** split: 198 plates (142 clinical, 56 smartphone photos). Each method counted the colonies on every plate and was compared with the human count. The most accurate shipped model is **RF-DETR-nano** (median error 0.75% on plates with 30-300 colonies); the fastest is **YOLOX-Tiny** (1.3 s per plate on CPU). Every shipped model is far more accurate than the classic baseline, OpenCFU (9.76%).

Models marked **(ships)** are the three built into colonycount v0.1. The YOLOv8 models are benchmark-only (AGPL-3.0) and OpenCFU is the classic non-deep-learning baseline.

> These are val-split numbers. The detection thresholds and the best training epochs were chosen on this split, so they are slightly optimistic; the held-out test split is run once on all models at release and this report is then regenerated.

## Headline: crop (default)

All plates. MdAPE is on plates with 30-300 true colonies; bias is predicted minus true.

| model | MdAPE (30-300) | MAE | bias | precision | recall | CPU s/plate | threshold | license | params |
|---|---|---|---|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | 0.75% | 3.5 | -0.2 | 97.4% | 97.2% | 8.7 | 0.4 | Apache-2.0 | 27.2 M |
| **YOLOX-Tiny** (ships) | 0.96% | 4.3 | -1.5 | 97.5% | 96.0% | 1.3 | 0.5 | Apache-2.0 | 5.0 M |
| **YOLOX-M** (ships) | 1.05% | 4.4 | -0.5 | 97.1% | 96.6% | 4.5 | 0.35 | Apache-2.0 | 25.3 M |
| YOLOv8-n | 1.38% | 7.7 | -3.4 | 96.4% | 93.0% | 0.9 | 0.4 | AGPL-3.0 | 3.1 M |
| YOLOv8-m | 1.21% | 7.4 | -2.9 | 96.6% | 93.7% | 5.2 | 0.35 | AGPL-3.0 | 25.8 M |
| OpenCFU | 9.76% | 22.5 | -11.8 | 83.9% | 73.5% | 0.6 | - | GPL-3.0 (tool) | - |

Counting defaults to the plate crop: `colonycount.count` finds the plate rim and counts inside it. 

## Headline: whole image

All plates. MdAPE is on plates with 30-300 true colonies; bias is predicted minus true.

| model | MdAPE (30-300) | MAE | bias | precision | recall | CPU s/plate | threshold | license | params |
|---|---|---|---|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | 1.18% | 3.8 | +1.0 | 96.7% | 97.8% | 14.4 | 0.35 | Apache-2.0 | 27.2 M |
| **YOLOX-Tiny** (ships) | 1.18% | 4.5 | -1.0 | 97.1% | 96.1% | 2.1 | 0.5 | Apache-2.0 | 5.0 M |
| **YOLOX-M** (ships) | 1.11% | 5.6 | -2.0 | 97.1% | 95.0% | 7.5 | 0.45 | Apache-2.0 | 25.3 M |
| YOLOv8-n | 1.43% | 7.2 | -0.7 | 95.7% | 95.0% | 1.5 | 0.35 | AGPL-3.0 | 3.1 M |
| YOLOv8-m | 1.38% | 7.9 | -2.3 | 96.1% | 93.7% | 8.7 | 0.35 | AGPL-3.0 | 25.8 M |
| OpenCFU | 8.57% | 22.5 | -9.3 | 84.9% | 76.5% | 0.6 | - | GPL-3.0 (tool) | - |

## By source

Clinical plates vs smartphone photos.

| model | mode | clinical MdAPE | clinical MAE | smartphone MdAPE | smartphone MAE |
|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | crop | 0.73% | 1.3 | 1.38% | 9.2 |
| **YOLOX-Tiny** (ships) | crop | 0.85% | 1.5 | 1.39% | 11.4 |
| **YOLOX-M** (ships) | crop | 0.85% | 1.5 | 2.06% | 12.0 |
| YOLOv8-n | crop | 1.05% | 2.5 | 2.00% | 21.1 |
| YOLOv8-m | crop | 1.00% | 2.5 | 1.45% | 19.8 |
| OpenCFU | crop | 9.70% | 11.1 | 13.33% | 51.6 |
| **RF-DETR-nano** (ships) | whole image | 1.17% | 1.9 | 1.42% | 8.7 |
| **YOLOX-Tiny** (ships) | whole image | 1.15% | 1.9 | 1.39% | 11.1 |
| **YOLOX-M** (ships) | whole image | 0.82% | 1.7 | 1.44% | 15.4 |
| YOLOv8-n | whole image | 1.55% | 3.4 | 1.40% | 16.8 |
| YOLOv8-m | whole image | 1.30% | 3.2 | 1.45% | 19.8 |
| OpenCFU | whole image | 9.57% | 13.1 | 5.56% | 46.3 |

## By colony density

True colony count per plate: <30 (46 plates), 30-300 (140 plates), >300 (12 plates). Plates under 30 colonies are TFTC (too few to count) and plates over 300 are TNTC (too numerous to count); lab practice counts only 30-300, which is why the headline MdAPE uses that range. Here MdAPE covers all plates in the bucket.

| model | mode | <30 MAE | <30 MdAPE | 30-300 MAE | 30-300 MdAPE | >300 MAE | >300 MdAPE |
|---|---|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | crop | 1.5 | 2.17% | 2.6 | 0.75% | 22.4 | 2.30% |
| **YOLOX-Tiny** (ships) | crop | 1.4 | 0.00% | 3.1 | 0.96% | 29.4 | 2.78% |
| **YOLOX-M** (ships) | crop | 1.5 | 0.00% | 3.4 | 1.05% | 28.7 | 3.56% |
| YOLOv8-n | crop | 3.2 | 6.11% | 5.4 | 1.38% | 52.0 | 4.05% |
| YOLOv8-m | crop | 3.2 | 8.54% | 5.0 | 1.21% | 50.6 | 4.47% |
| OpenCFU | crop | 9.3 | 31.67% | 16.4 | 9.76% | 145.0 | 19.07% |
| **RF-DETR-nano** (ships) | whole image | 1.6 | 4.58% | 3.0 | 1.18% | 21.8 | 3.31% |
| **YOLOX-Tiny** (ships) | whole image | 1.4 | 0.00% | 3.4 | 1.18% | 28.5 | 2.40% |
| **YOLOX-M** (ships) | whole image | 1.5 | 2.08% | 4.0 | 1.11% | 39.6 | 3.91% |
| YOLOv8-n | whole image | 3.8 | 5.83% | 5.2 | 1.43% | 42.9 | 4.36% |
| YOLOv8-m | whole image | 3.8 | 7.33% | 5.6 | 1.38% | 50.3 | 4.39% |
| OpenCFU | whole image | 10.3 | 31.88% | 16.9 | 8.57% | 134.5 | 19.38% |

## CPU speed

Median seconds per plate on CPU, one plate at a time (OpenCFU runs in a container on images resized to 1000 px).

| model | crop | whole image |
|---|---|---|
| **RF-DETR-nano** (ships) | 8.7 | 14.4 |
| **YOLOX-Tiny** (ships) | 1.3 | 2.1 |
| **YOLOX-M** (ships) | 4.5 | 7.5 |
| YOLOv8-n | 0.9 | 1.5 |
| YOLOv8-m | 5.2 | 8.7 |
| OpenCFU | 0.6 | 0.6 |

## How the threshold was chosen

A detection counts when its score is at least the threshold. For each model the threshold was swept automatically on the validation split from 0.20 to 0.80 and the value with the lowest MdAPE (30-300) was kept, ties broken by the lowest MAE. MdAPE by threshold (the chosen one in bold):

| threshold | RF-DETR-nano crop | RF-DETR-nano whole image | YOLOX-Tiny crop | YOLOX-Tiny whole image | YOLOX-M crop | YOLOX-M whole image |
|---|---|---|---|---|---|---|
| 0.2 | 1.40% | 1.83% | 1.48% | 1.87% | 1.40% | 1.48% |
| 0.25 | 1.30% | 1.51% | 1.41% | 1.67% | 1.27% | 1.38% |
| 0.3 | 1.16% | 1.26% | 1.33% | 1.44% | 1.21% | 1.27% |
| 0.35 | 0.84% | **1.18%** | 1.14% | 1.39% | **1.05%** | 1.19% |
| 0.4 | **0.75%** | 1.18% | 1.14% | 1.39% | 1.11% | 1.29% |
| 0.45 | 1.01% | 1.18% | 1.14% | 1.38% | 1.05% | **1.11%** |
| 0.5 | 1.18% | 1.28% | **0.96%** | **1.18%** | 1.17% | 1.26% |
| 0.55 | 1.37% | 1.36% | 1.04% | 1.25% | 1.05% | 1.29% |
| 0.6 | 1.44% | 1.46% | 1.18% | 1.33% | 1.18% | 1.35% |
| 0.65 | 2.08% | 2.01% | 1.61% | 1.89% | 1.53% | 1.72% |
| 0.7 | 2.94% | 2.83% | 2.27% | 2.19% | 1.83% | 2.15% |
| 0.75 | 4.20% | 4.20% | 2.90% | 2.80% | 2.53% | 2.48% |
| 0.8 | 6.13% | 5.56% | 4.20% | 4.04% | 3.51% | 4.06% |

## How to read these metrics

- **MdAPE (30-300)**: median absolute percentage error of the plate count, on plates with 30-300 true colonies. Half of the plates are off by less than this.
- **MAE**: mean absolute error, in colonies per plate (all plates).
- **bias**: mean of predicted minus true count; positive means over-counting.
- **precision**: share of detections that are real colonies. **recall**: share of real colonies that were found.
- **match radius**: a detection matches a true colony when its centre is within half the colony's mean box side of the true centre; each colony matches at most once.
<!-- results:end -->

## Settings (frozen 2026-11-15)

These numbers use the settings frozen before the one-time test run, all chosen on val:

| setting | value | was (results of 2026-11-13/14) |
|---|---|---|
| plate finder | Hough circle on the brightest colour channel, edge threshold 50 | grey image, edge threshold 100 (missed 55 of 198 plates) |
| plate radius pad (`COLONYCOUNT_PLATE_PAD`) | +5% | none (cut off 1.35% of colonies at the rim) |
| default mode | plate crop | plate crop |
| NMS IoU (`COLONYCOUNT_NMS_IOU`) | 0.4 | 0.5 |
| score threshold, crop / whole image | RF-DETR-nano 0.40 / 0.35, YOLOX-Tiny 0.50 / 0.50, YOLOX-M 0.35 / 0.45, YOLOv8-n 0.40 / 0.35, YOLOv8-m 0.35 / 0.35 | 0.30-0.50, picked under the old plate finder and NMS 0.5 |
| OpenCFU setting, crop / whole image | `-d bil -t 5 -r 4 -R 30` at 1000 px / `-d bil -t 5 -r 6 -R 45` at 1500 px (full 56-setting grid) | partial grid (see below) |

Thresholds come from the automatic val sweep (0.20-0.80, lowest MdAPE 30-300, tie-break MAE). Since
the plate finder, pad, NMS and thresholds were all chosen on these 198 plates, val is now more
optimistic than before: the honest numbers come from the one-time test run, which reuses these
settings unchanged (`bench/evaluate.py --split test` requires a fixed `--threshold`;
`opencfu_runner.py --split test` requires the val-tuned `--setting`). The previous val outputs are
kept outside the repo under `<data dir>/bench/_pre-wave2/`.

## Files (`benchmark/val/`)

| file | what |
|---|---|
| `rfdetr-nano.json` / `.md` | colonycount with RF-DETR-nano, plate crop on (`bench/evaluate.py`) |
| `rfdetr-nano-noplate.json` / `.md` | same, whole image (`--no-plate`) |
| `yolox-tiny.json` / `.md`, `-noplate` | YOLOX-Tiny, crop / whole image |
| `yolox-m.json` / `.md`, `-noplate` | YOLOX-M, crop / whole image |
| `yolov8n.json` / `.md`, `-noplate` | YOLOv8-n (benchmark only), crop / whole image |
| `yolov8m.json` / `.md`, `-noplate` | YOLOv8-m (benchmark only), crop / whole image |
| `opencfu.json` / `.md`, `-noplate` | OpenCFU baseline at its best tuned setting (`bench/opencfu_runner.py`) |
| `opencfu-grid.jsonl`, `opencfu-noplate-grid.jsonl` | the full OpenCFU settings grid on val (56 settings each), one JSON line per setting |
| `comparison.md` | all methods side by side, per source, per density bucket and overall (`bench/compare.py`) |

Columns: `mae` and `bias` (predicted minus true) in colonies per plate; `mdape_30_300` is the
median absolute percentage error on plates whose true count is 30-300 (`n_30_300` of them);
`precision`/`recall` are per colony; `plate_found` is the share of plates where the plate rim was
found (0 with the crop off). CPU s/plate is the median wall time per plate on CPU, one plate at a
time: around `colonycount.count` for the models, around the `opencfu` call inside the container for
OpenCFU (which sees images resized to the setting's longest side).

The JSONs are the tools' outputs with two changes made by `bench/publish_results.py`: paths are
relative to `COLONYCOUNT_DATA_DIR` (none are absolute), and `cpu_s_per_plate` (median/mean/max,
from the `.md`'s CPU-time line, since the JSONs are kept free of timings) is added.

## Commands

Equivalent to the runs above (the output folders match what `publish_results.py` reads):

```powershell
$d = $env:COLONYCOUNT_DATA_DIR
# each model: crop, then whole image; writes bench/<name>/val[-noplate].json/.md (+ .npz with --save-detections)
foreach ($m in @(@('rfdetr-nano','public-rfdetr-nano-20261113'), @('yolox_tiny','yolox_tiny-20261114'),
                 @('yolox_m','yolox_m-20261114'), @('yolov8n','yolov8n-20261113'), @('yolov8m','yolov8m-20261113'))) {
  uv run python bench/evaluate.py --model "$d/models/$($m[1]).onnx" --split val --save-detections --out "$d/bench/$($m[0])"
  uv run python bench/evaluate.py --model "$d/models/$($m[1]).onnx" --split val --save-detections --no-plate --out "$d/bench/$($m[0])"
}
# OpenCFU: tune the grid on val (writes bench/opencfu/opencfu-val[-noplate].json/.md and -grid.json)
docker build -t colonycount-opencfu bench/opencfu
uv run python bench/opencfu_runner.py --split val --tune --out "$d/bench/opencfu"
uv run python bench/opencfu_runner.py --split val --tune --no-plate --out "$d/bench/opencfu"
# copy into results/ and regenerate the report, comparison.md and the README summary
uv run python bench/publish_results.py --split val
```

(POSIX shells: `$COLONYCOUNT_DATA_DIR`; the default data dir is `~/data/colonycount`.)

## Provenance

- Date: 2026-11-15 (val re-run under the frozen settings). Split: **val**, 198 plates (142 clinical,
  56 smartphone). The test split was not touched; it is reserved for one run on all models with the
  settings above.
- Benchmark code: `bench/evaluate.py` and `bench/opencfu_runner.py` as in the commit that published
  these files. The runs were sequential, in one PowerShell window, with nothing else heavy running,
  so the CPU times are comparable across methods.
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
- Shipped in v0.1 (owner decision 2026-11-15): `rfdetr-nano` (default), `yolox-tiny` and `yolox-m`.
  YOLOv8-n/m are benchmark-only (AGPL-3.0).
- OpenCFU 4.0.0 (Debian package `opencfu 4.0.0+dfsg-2+b2`), image from `bench/opencfu/Dockerfile`.
- Matching rule (`bench/evaluate.py`): a prediction matches a true colony when its centre lies
  within half the colony box's mean side of the box centre; greedy one-to-one, nearest pairs first.
