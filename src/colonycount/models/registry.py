"""Registry of built-in models: download from the Hugging Face Hub, SHA-256 check, local cache.

Set ``COLONYCOUNT_MODEL_PATH`` (or pass ``model_path=``) to use a local ONNX file instead.
"""

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from colonycount._env import env_path


@dataclass(frozen=True)
class Model:
    name: str
    version: str
    license: str
    url: str
    sha256: str | None  # None = not published yet


# Adding a model (e.g. `agar`, task 6.4) is one more entry here.
# ponytail: `public` is a placeholder until task 5.2 publishes the weights and pins the checksum.
MODELS = {
    m.name: m
    for m in [
        Model(
            name="public",
            version="0.1.0",
            license="CC-BY-4.0",
            url="https://huggingface.co/rafali-azz/colonycount-public/resolve/v0.1.0/public.onnx",
            sha256=None,
        ),
    ]
}


def get(name: str) -> Model:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; available models: {', '.join(sorted(MODELS))}"
        ) from None


def resolve(name: str, model_path: str | os.PathLike | None = None) -> tuple[Path, str]:
    """Return (path to the ONNX file, model version), downloading and caching it if needed.

    ``model_path`` defaults to ``COLONYCOUNT_MODEL_PATH``; the cache is ``COLONYCOUNT_CACHE_DIR``
    or the platform's user cache directory.
    """
    model = get(name)
    local = model_path or env_path("COLONYCOUNT_MODEL_PATH", None)
    if local:
        if not Path(local).is_file():
            raise FileNotFoundError(f"model file not found: {local}")
        return Path(local), "local"
    if model.sha256 is None:
        raise RuntimeError(
            f"model {name!r} is not published yet; set COLONYCOUNT_MODEL_PATH or pass model_path= "
            "to use a local ONNX file"
        )
    cache = env_path("COLONYCOUNT_CACHE_DIR", None) or platformdirs.user_cache_path("colonycount")
    target = cache / f"{model.name}-{model.version}.onnx"
    if not target.is_file():
        _download(model, target)
    return target, model.version


def _download(model: Model, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(model.url, timeout=60) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                digest.update(chunk)
                f.write(chunk)
    except OSError as e:  # URLError is an OSError
        part.unlink(missing_ok=True)
        raise RuntimeError(f"could not download model {model.name!r} from {model.url}: {e}") from e
    if digest.hexdigest() != model.sha256:
        part.unlink()
        raise RuntimeError(
            f"checksum mismatch for model {model.name!r}: expected {model.sha256}, "
            f"got {digest.hexdigest()}"
        )
    os.replace(part, target)  # only verified files ever land in the cache
