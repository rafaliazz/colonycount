# Benchmark results

Plate-count benchmark outputs, committed so they can be read without the data. Not part of the
package (only `src/` goes into the wheel). Start with
[`benchmark/val/comparison.md`](benchmark/val/comparison.md).

## Files (`benchmark/val/`)

| file | what |
|---|---|
| `rfdetr-nano.json` / `.md` | colonycount with the RF-DETR-nano model, plate crop on, threshold 0.45 (`bench/evaluate.py`) |
| `rfdetr-nano-noplate.json` / `.md` | same, whole image (`--no-plate`) |
| `yolov8n.json` / `.md` | colonycount with the YOLOv8-n model, plate crop on, threshold 0.3 (`bench/evaluate.py`) |
| `yolov8n-noplate.json` / `.md` | same, whole image, threshold 0.35 |
| `yolov8m.json` / `.md` | colonycount with the YOLOv8-m model, plate crop on, threshold 0.3 (`bench/evaluate.py`) |
| `yolov8m-noplate.json` / `.md` | same, whole image, threshold 0.35 |
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
  not touched.
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
- OpenCFU 4.0.0 (Debian package `opencfu 4.0.0+dfsg-2+b2`), image from `bench/opencfu/Dockerfile`.
- Threshold 0.45 was chosen **on val** (lowest MdAPE 30-300, tie-break MAE), and OpenCFU's settings
  were also tuned on val, so all numbers here are optimistic; an unbiased number needs the test
  split.
- Matching rule (`bench/evaluate.py`): a prediction matches a true colony when its centre lies
  within half the colony box's mean side of the box centre; greedy one-to-one, nearest pairs first.
- Determinism: the RF-DETR crop run was repeated (`run2`) and `val.json`/`val.md` were
  byte-identical to `run1`.

## Caveats

- CPU s/plate is not strictly comparable between runs: the YOLOv8-m whole-image run was done
  from the owner's terminal, while the YOLOv8-m crop run was done from a Claude Code background
  shell under memory pressure, which is part of why its CPU time per plate (9.48 s) is higher
  than the whole-image run's (8.23 s).
- The OpenCFU grid (2 sides x 7 threshold modes x 2 x 2 radius ranges) was stopped early. Plate
  crop: 27 of the 28 settings at 1000 px (`-a -r 4 -R 60` missing), none at 1500 px. Whole image:
  all 28 at 1000 px, 2 of 28 at 1500 px. The final OpenCFU runs use the best 1000 px setting of
  each grid (crop `-d bil -t 5 -r 4 -R 60`, whole image `-d bil -t 3 -r 4 -R 30`).
