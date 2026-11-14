"""Read ``COLONYCOUNT_*`` configuration from the environment where it is used.

Each helper reads ``os.getenv(name)`` when called; unset or empty means ``default``. A value that
does not parse raises ValueError naming the variable. Entry points (the CLI, train scripts) load
``.env`` from the working directory first with ``dotenv.load_dotenv``, which never overrides real
environment variables; the library itself does not. Every variable is listed in ``.env.dist``.
"""

import os
from pathlib import Path


def _read(name, default, parse):
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return parse(raw)
    except ValueError as e:
        raise ValueError(f"{name}={raw!r} is invalid: {e}") from None


def _in_range(value, lo, hi):
    if (lo is not None and value < lo) or (hi is not None and value > hi) or value != value:
        raise ValueError(f"must be in [{lo}, {hi}]")
    return value


def env_str(name: str, default: str) -> str:
    return _read(name, default, str)


def env_int(name: str, default: int, lo: int | None = None) -> int:
    return _read(name, default, lambda s: _in_range(int(s), lo, None))


def env_float(
    name: str, default: float | None, lo: float | None = None, hi: float | None = None
) -> float | None:
    return _read(name, default, lambda s: _in_range(float(s), lo, hi))


def env_path(name: str, default: str | None) -> Path | None:
    value = _read(name, default, str)
    return None if value is None else Path(value).expanduser()


def env_tuple(name: str, default: tuple) -> tuple:
    """Comma-separated numbers, as many as ``default`` has and of the same types."""

    def parse(s):
        parts = s.split(",")
        if len(parts) != len(default):
            raise ValueError(f"expected {len(default)} comma-separated numbers")
        return tuple(type(d)(p) for d, p in zip(default, parts, strict=True))

    return _read(name, default, parse)
