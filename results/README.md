# Benchmark results

Plate-count benchmark outputs, committed so they can be read without the data. Not part of the
package (only `src/` goes into the wheel). The report below is
generated from the benchmark JSON; the full per-method tables are in
[`benchmark/val/comparison.md`](benchmark/val/comparison.md).

<!-- results:begin -->
## Summary

This is the **test** split: 192 plates (142 clinical, 50 smartphone photos). Each method counted the colonies on every plate and was compared with the human count. The most accurate shipped model is **YOLOX-M** (median error 0.92% on plates with 30-300 colonies); the fastest is **YOLOX-Tiny** (1.6 s per plate on CPU). Every shipped model is far more accurate than the classic baseline, OpenCFU (10.12%).

Models marked **(ships)** are the three built into colonycount v0.1. The YOLOv8 models are benchmark-only (AGPL-3.0) and OpenCFU is the classic non-deep-learning baseline.

> Held-out test split, run once; thresholds were fixed on the validation split.

## Headline: crop (default)

All plates. MdAPE is on plates with 30-300 true colonies; bias is predicted minus true.

| model | MdAPE (30-300) | MAE | bias | precision | recall | CPU s/plate | threshold | license | params |
|---|---|---|---|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | 1.06% | 1.1 | -0.0 | 98.3% | 98.2% | 9.0 | 0.4 | Apache-2.0 | 27.2 M |
| **YOLOX-Tiny** (ships) | 1.45% | 1.8 | -1.0 | 98.5% | 97.1% | 1.6 | 0.5 | Apache-2.0 | 5.0 M |
| **YOLOX-M** (ships) | 0.92% | 1.0 | +0.0 | 98.1% | 98.1% | 6.0 | 0.35 | Apache-2.0 | 25.3 M |
| YOLOv8-n | 2.11% | 2.4 | -1.3 | 98.3% | 96.3% | 1.1 | 0.4 | AGPL-3.0 | 3.1 M |
| YOLOv8-m | 1.53% | 2.2 | -1.1 | 98.3% | 96.7% | 6.6 | 0.35 | AGPL-3.0 | 25.8 M |
| OpenCFU | 10.12% | 10.6 | +4.2 | 82.7% | 87.9% | 1.1 | - | GPL-3.0 (tool) | - |

Counting defaults to the plate crop: `colonycount.count` finds the plate rim and counts inside it. 

## Headline: whole image

All plates. MdAPE is on plates with 30-300 true colonies; bias is predicted minus true.

| model | MdAPE (30-300) | MAE | bias | precision | recall | CPU s/plate | threshold | license | params |
|---|---|---|---|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | 1.45% | 1.2 | +0.6 | 97.7% | 98.6% | 17.4 | 0.35 | Apache-2.0 | 27.2 M |
| **YOLOX-Tiny** (ships) | 1.59% | 1.9 | -0.9 | 98.4% | 97.1% | 2.6 | 0.5 | Apache-2.0 | 5.0 M |
| **YOLOX-M** (ships) | 1.16% | 1.4 | -0.5 | 98.3% | 97.5% | 9.9 | 0.45 | Apache-2.0 | 25.3 M |
| YOLOv8-n | 1.65% | 2.0 | -0.7 | 98.0% | 96.9% | 1.8 | 0.35 | AGPL-3.0 | 3.1 M |
| YOLOv8-m | 1.61% | 2.3 | -1.1 | 98.3% | 96.7% | 11.0 | 0.35 | AGPL-3.0 | 25.8 M |
| OpenCFU | 7.95% | 9.9 | +4.8 | 84.4% | 90.6% | 1.0 | - | GPL-3.0 (tool) | - |

## By source

Clinical plates vs smartphone photos.

| model | mode | clinical MdAPE | clinical MAE | smartphone MdAPE | smartphone MAE |
|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | crop | 0.57% | 0.7 | 1.85% | 2.0 |
| **YOLOX-Tiny** (ships) | crop | 0.46% | 0.7 | 4.42% | 5.1 |
| **YOLOX-M** (ships) | crop | 0.58% | 0.7 | 1.44% | 2.0 |
| YOLOv8-n | crop | 0.99% | 1.0 | 5.72% | 6.4 |
| YOLOv8-m | crop | 0.99% | 1.0 | 5.13% | 5.8 |
| OpenCFU | crop | 9.73% | 8.8 | 13.29% | 15.7 |
| **RF-DETR-nano** (ships) | whole image | 1.16% | 0.9 | 1.81% | 2.2 |
| **YOLOX-Tiny** (ships) | whole image | 0.80% | 0.7 | 4.42% | 5.1 |
| **YOLOX-M** (ships) | whole image | 0.61% | 0.8 | 2.54% | 3.3 |
| YOLOv8-n | whole image | 0.94% | 1.0 | 4.56% | 4.8 |
| YOLOv8-m | whole image | 1.23% | 1.0 | 5.55% | 5.9 |
| OpenCFU | whole image | 6.58% | 5.6 | 18.79% | 21.9 |

## By colony density

True colony count per plate: <30 (78 plates), 30-300 (114 plates), >300 (0 plates). Plates under 30 colonies are TFTC (too few to count) and plates over 300 are TNTC (too numerous to count); lab practice counts only 30-300, which is why the headline MdAPE uses that range. Here MdAPE covers all plates in the bucket.

| model | mode | <30 MAE | <30 MdAPE | 30-300 MAE | 30-300 MdAPE | >300 MAE | >300 MdAPE |
|---|---|---|---|---|---|---|---|
| **RF-DETR-nano** (ships) | crop | 0.5 | 0.00% | 1.4 | 1.06% | - | - |
| **YOLOX-Tiny** (ships) | crop | 0.5 | 0.00% | 2.7 | 1.45% | - | - |
| **YOLOX-M** (ships) | crop | 0.5 | 0.00% | 1.4 | 0.92% | - | - |
| YOLOv8-n | crop | 0.8 | 0.00% | 3.4 | 2.11% | - | - |
| YOLOv8-m | crop | 0.7 | 0.00% | 3.3 | 1.53% | - | - |
| OpenCFU | crop | 7.0 | 30.07% | 13.1 | 10.12% | - | - |
| **RF-DETR-nano** (ships) | whole image | 0.6 | 0.00% | 1.6 | 1.45% | - | - |
| **YOLOX-Tiny** (ships) | whole image | 0.6 | 0.00% | 2.7 | 1.59% | - | - |
| **YOLOX-M** (ships) | whole image | 0.6 | 0.00% | 2.0 | 1.16% | - | - |
| YOLOv8-n | whole image | 0.8 | 0.00% | 2.8 | 1.65% | - | - |
| YOLOv8-m | whole image | 0.8 | 0.00% | 3.4 | 1.61% | - | - |
| OpenCFU | whole image | 4.1 | 21.06% | 13.8 | 7.95% | - | - |

## CPU speed

Median seconds per plate on CPU, one plate at a time (OpenCFU runs in a container on images resized to 1000 px).

| model | crop | whole image |
|---|---|---|
| **RF-DETR-nano** (ships) | 9.0 | 17.4 |
| **YOLOX-Tiny** (ships) | 1.6 | 2.6 |
| **YOLOX-M** (ships) | 6.0 | 9.9 |
| YOLOv8-n | 1.1 | 1.8 |
| YOLOv8-m | 6.6 | 11.0 |
| OpenCFU | 1.1 | 1.0 |

## How the threshold was chosen

A detection counts when its score is at least the threshold. For each model the threshold was swept automatically on the validation split from 0.20 to 0.80 and the value with the lowest MdAPE (30-300) was kept, ties broken by the lowest MAE. Below, MdAPE on this test split by threshold, for information only: the bold value is the one fixed on validation, not the best one here.

| threshold | RF-DETR-nano crop | RF-DETR-nano whole image | YOLOX-Tiny crop | YOLOX-Tiny whole image | YOLOX-M crop | YOLOX-M whole image |
|---|---|---|---|---|---|---|
| 0.2 | 2.34% | 2.33% | 1.58% | 1.63% | 1.09% | 1.10% |
| 0.25 | 2.03% | 1.93% | 1.45% | 1.47% | 0.91% | 0.94% |
| 0.3 | 1.59% | 1.64% | 1.47% | 1.50% | 0.91% | 0.89% |
| 0.35 | 1.10% | **1.45%** | 1.16% | 1.53% | **0.92%** | 0.98% |
| 0.4 | **1.06%** | 1.34% | 1.45% | 1.65% | 0.99% | 1.12% |
| 0.45 | 1.35% | 1.40% | 1.34% | 1.63% | 1.41% | **1.16%** |
| 0.5 | 1.43% | 1.45% | **1.45%** | **1.59%** | 1.41% | 1.17% |
| 0.55 | 2.08% | 2.15% | 1.39% | 1.59% | 1.53% | 1.45% |
| 0.6 | 2.31% | 2.31% | 1.48% | 1.86% | 1.54% | 1.51% |
| 0.65 | 2.92% | 2.78% | 2.00% | 1.80% | 1.45% | 1.45% |
| 0.7 | 4.58% | 4.29% | 2.46% | 2.31% | 2.12% | 2.10% |
| 0.75 | 7.20% | 6.56% | 2.73% | 2.69% | 3.15% | 3.03% |
| 0.8 | 9.16% | 9.71% | 4.63% | 5.29% | 6.16% | 4.63% |

## How to read these metrics

- **MdAPE (30-300)**: median absolute percentage error of the plate count, on plates with 30-300 true colonies. Half of the plates are off by less than this.
- **MAE**: mean absolute error, in colonies per plate (all plates).
- **bias**: mean of predicted minus true count; positive means over-counting.
- **precision**: share of detections that are real colonies. **recall**: share of real colonies that were found.
- **match radius**: a detection matches a true colony when its centre is within half the colony's mean box side of the true centre; each colony matches at most once.
<!-- results:end -->

## Test run (2026-11-16)

The report above is the **test** split: 192 plates (142 clinical, 50 smartphone) that were never
used for training or for any choice. It was run **once**, on 2026-11-15/16, for all methods in both
modes, with every setting frozen on val beforehand (see "Settings" below): `bench/evaluate.py
--split test --threshold <frozen>` for the models and `bench/opencfu_runner.py --split test
--setting <val-tuned>` for OpenCFU (both refuse the test split without them). The val outputs are in
`benchmark/val/` and stay as they are.

- **Release gate** (MdAPE 30-300 at most 5% on test, crop mode): RF-DETR-nano 1.06%, YOLOX-Tiny
  1.45%, YOLOX-M 0.92%; all three pass.
- **Default model:** `rfdetr-nano` stays the default (decided on val before the test run). YOLOX-M is
  slightly ahead on test (0.92% vs 1.06% MdAPE, 1.04 vs 1.06 MAE), a difference within the noise of
  114 in-range plates; choosing the default from test numbers would reuse the test split for a
  decision.
- **YOLOX-Tiny on smartphone photos** is the weak spot: MdAPE 4.42%, undercounting by 4.5 colonies
  per plate (clinical 0.46%); RF-DETR-nano (1.85%) and YOLOX-M (1.44%) handle them much better.
- **MAE is not comparable between val and test:** the test split has no plate above 300 colonies
  (val: 12) and more near-empty plates (78 below 30, val: 46), so its MAE is lower. Compare MdAPE
  30-300 instead (val vs test, crop: RF-DETR-nano 0.75% / 1.06%, YOLOX-Tiny 0.96% / 1.45%, YOLOX-M
  1.05% / 0.92%, YOLOv8-n 1.38% / 2.11%, YOLOv8-m 1.21% / 1.53%, OpenCFU 9.76% / 10.12%).
- **CPU times on test are 20-40% higher than on val** for every method, likely from a warmer
  laptop overnight; the ratios between methods hold. Quote speed from the val run, where all
  methods ran back to back under the same conditions.

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
