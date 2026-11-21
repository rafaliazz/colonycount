# colonycount

Open-source, offline bacterial colony counter for agar plate photos: Python package, CLI and napari plugin, with a published benchmark against human counts.

Status: pre-alpha.

Datasets, licenses and attribution: see [DATA.md](DATA.md).

## Model results

<!-- results:begin -->
On the test split, plate crop on (the default). MdAPE is the median error of the plate count on plates with 30-300 colonies, MAE the mean error in colonies per plate, and CPU time is the median per plate:

| model | MdAPE (30-300) | MAE | CPU s/plate | license |
|---|---|---|---|---|
| **RF-DETR-nano** (ships) | 1.06% | 1.1 | 9.0 | Apache-2.0 |
| **YOLOX-Tiny** (ships) | 1.45% | 1.8 | 1.6 | Apache-2.0 |
| **YOLOX-M** (ships) | 0.92% | 1.0 | 6.0 | Apache-2.0 |
| OpenCFU | 10.12% | 10.6 | 1.1 | GPL-3.0 (tool) |

Held-out test split (unseen plates), run once with every setting fixed on the validation split. Full report: [results/README.md](results/README.md).
<!-- results:end -->

All per-method tables: [results/benchmark/val/comparison.md](results/benchmark/val/comparison.md).

## Quickstart

> The v0.1 weights (`rfdetr-nano` (default), `yolox-tiny` (fastest) and `yolox-m`) are on Hugging Face, [`rafai-azz/colonycount-public`](https://huggingface.co/rafai-azz/colonycount-public) (tag `v0.1.0`; the repo is public from the v0.1 release), and the package is not on PyPI yet.
> Until then, install from source. To use your own ONNX detector, point colonycount at it with `COLONYCOUNT_MODEL_PATH` (or `model_path=` / `--model-path`).
> The model's input/output contract is documented in `src/colonycount/detection/runner.py`.

Requires Python 3.14. On an older Python, get 3.14 with `uv python install 3.14` (no admin rights needed).

Install (once released):

```sh
pip install colonycount
```

Count one plate from Python:

```python test
import colonycount

print(colonycount.count("plate.jpg").total)
```

Count a folder of plates into a CSV with CFU/mL (dilution 1e-4, 0.1 mL plated). Glob patterns are expanded by colonycount itself, so this also works in Windows shells:

```sh test
colonycount plates/*.jpg --dilution 1e-4 --volume 0.1 -o counts.csv
colonycount plates/*.jpg --detections detections/ -o counts.csv
colonycount --list-models
```

`counts.csv` has one row per image: `file, count, plate_found, range_flag, threshold, model, model_version, package_version, cfu_per_ml, error`.
`range_flag` is `TFTC` below 30 colonies and `TNTC` above 300, `OK` in between (change with `--range LOW HIGH`); the raw count is always reported.
A failed image gets an `error` and the command exits non-zero; the other images are still counted.
Weights download once on first use, are checked against a pinned SHA-256 and cached; after that counting works offline, and images never leave your machine.

The code blocks tagged `test` above run in CI (`tests/test_readme.py`).

## Configuration

Settings are `COLONYCOUNT_*` environment variables; [`.env.dist`](.env.dist) lists every one with its default.
To change some, copy it to `.env` (git-ignored) and edit only what you need; secrets such as tokens go only in `.env` or CI secrets.

Precedence: CLI flag or function argument > environment variable > `.env` > default.
`.env` is read from the working directory by the CLI and the `train/` scripts; the Python library (`colonycount.count`) reads only real environment variables.

Each built-in model has its own default score threshold (`rfdetr-nano`: 0.40, `yolox-tiny`: 0.50, `yolox-m`: 0.35, frozen on the validation split), shown by `--list-models`. The threshold used is recorded in `counts.csv`. Precedence: `--threshold` > `COLONYCOUNT_THRESHOLD` > the model's default > 0.5 (a local `--model-path` ONNX file).

Main variables: `COLONYCOUNT_MODEL`, `COLONYCOUNT_MODEL_PATH`, `COLONYCOUNT_THRESHOLD`, `COLONYCOUNT_COUNTABLE_RANGE`, `COLONYCOUNT_TILE_OVERLAP`, `COLONYCOUNT_CACHE_DIR`, `COLONYCOUNT_DATA_DIR`, and the training ones (`COLONYCOUNT_TRAIN_TILE_SIZE`, `COLONYCOUNT_TRAIN_EPOCHS`, `COLONYCOUNT_SEED`, ...).
The inference tile size is not a setting: it is read from the ONNX model's input shape (`images` `[1, 3, T, T]`), and the model is checked against its contract when it loads.

## Development

Requires [uv](https://docs.astral.sh/uv/); it fetches Python 3.14 (pinned in `.python-version`) if you don't have it.

```sh
uv sync                 # core + dev tools
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pre-commit install   # hooks: ruff on commit; lock check + pytest on push
```

CI lints, builds the package (wheel + sdist, install-tested) and runs tests on Linux, Windows and macOS, all on Python 3.14.

Optional extras: `uv sync --extra napari`, `--extra demo`, `--extra train`.
