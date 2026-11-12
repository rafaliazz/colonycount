# colonycount

Open-source, offline bacterial colony counter for agar plate photos: Python package, CLI and napari plugin, with a published benchmark against human counts.

Status: pre-alpha.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                 # core + dev tools
uv run pytest
uv run ruff check && uv run ruff format --check
```

Optional extras: `uv sync --extra napari`, `--extra demo`, `--extra train`.
