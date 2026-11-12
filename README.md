# colonycount

Open-source, offline bacterial colony counter for agar plate photos: Python package, CLI and napari plugin, with a published benchmark against human counts.

Status: pre-alpha.

Datasets, licenses and attribution: see [DATA.md](DATA.md).

## Quickstart

> The built-in `public` model weights are **not published yet** (planned for v0.1), and the package is not on PyPI yet.
> Until then, install from source and point colonycount at a local ONNX detector with `COLONYCOUNT_MODEL_PATH` (or `model_path=` / `--model-path`).
> The model's input/output contract is documented in `src/colonycount/detect.py`.

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

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                 # core + dev tools
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pre-commit install   # hooks: ruff on commit; lock check + pytest on push
```

CI lints, builds the package (wheel + sdist, install-tested) and runs tests on Linux, Windows and macOS.

Optional extras: `uv sync --extra napari`, `--extra demo`, `--extra train`.
