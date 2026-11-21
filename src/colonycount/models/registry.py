"""Registry of built-in models: download from the Hugging Face Hub, SHA-256 check, local cache.

Set ``COLONYCOUNT_MODEL_PATH`` (or pass ``model_path=``) to use a local ONNX file instead.
"""

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from colonycount._env import env_float, env_path


@dataclass(frozen=True)
class Model:
    name: str
    version: str
    license: str
    url: str
    sha256: str | None  # None = not published yet
    threshold: float  # default score threshold, chosen on val


# Adding a model (e.g. `agar`, task 6.4) is one more entry here.
# Thresholds frozen on val 2026-11-15, see results/README.md.
MODELS = {
    m.name: m
    for m in [
        Model(
            name="rfdetr-nano",
            version="0.1.0",
            license="CC-BY-4.0",
            url="https://huggingface.co/rafai-azz/colonycount-public/resolve/v0.1.0/rfdetr-nano.onnx",
            sha256="42e7602cbcbb4b18d1851464b660a42d744d0b86edf11c02eb62046141081586",
            threshold=0.40,
        ),
        Model(
            name="yolox-tiny",
            version="0.1.0",
            license="CC-BY-4.0",
            url="https://huggingface.co/rafai-azz/colonycount-public/resolve/v0.1.0/yolox-tiny.onnx",
            sha256="cbc98f010e242726c781b0a4c6d6ea5a6d6c2b81ed7f28b2b30603ccacf8900e",
            threshold=0.50,
        ),
        Model(
            name="yolox-m",
            version="0.1.0",
            license="CC-BY-4.0",
            url="https://huggingface.co/rafai-azz/colonycount-public/resolve/v0.1.0/yolox-m.onnx",
            sha256="392a148324c462b36777d353ded9be3db17a67f9adbecab376099183ff4500f6",
            threshold=0.35,
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


def threshold(name: str, model_path: str | os.PathLike | None = None) -> float:
    """Score threshold: ``COLONYCOUNT_THRESHOLD``, else the model's default, else 0.5.

    A local ONNX file (``model_path`` or ``COLONYCOUNT_MODEL_PATH``) is not a registry model, so it
    gets the global 0.5.
    """
    local = model_path or env_path("COLONYCOUNT_MODEL_PATH", None)
    env = env_float("COLONYCOUNT_THRESHOLD", None, 0, 1)
    return env if env is not None else 0.5 if local else get(name).threshold


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
