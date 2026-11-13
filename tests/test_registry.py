"""Model registry, download, checksum, cache and offline scenarios."""

import hashlib
import urllib.error
import urllib.request

import pytest
from conftest import make_plate

import colonycount
from colonycount.models import registry


def offline(*args, **kwargs):
    raise urllib.error.URLError("network is unreachable")


@pytest.fixture
def published(model_file, monkeypatch, tmp_path):
    """Registry whose models are 'published' at a local file:// URL; records download calls."""
    sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
    url = model_file.as_uri()
    monkeypatch.delenv("COLONYCOUNT_MODEL_PATH", raising=False)
    monkeypatch.setenv("COLONYCOUNT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        registry,
        "MODELS",
        {
            "public": registry.Model("public", "9.9", "CC-BY-4.0", url, sha),
            "agar": registry.Model("agar", "2.0", "CC-BY-NC-2.0", url, sha),
        },
    )
    calls = []
    real = urllib.request.urlopen

    def urlopen(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def test_first_run_downloads_verifies_and_caches(published, tmp_path):
    path, version = registry.resolve("public")
    assert path.parent == tmp_path / "cache" and version == "9.9"
    assert registry.resolve("public")[0] == path
    assert len(published) == 1  # downloaded once
    assert colonycount.count(make_plate()[0]).model_version == "9.9"


def test_no_network_with_cache(published, monkeypatch):
    registry.resolve("public")
    monkeypatch.setattr(urllib.request, "urlopen", offline)
    img, truth = make_plate()
    assert colonycount.count(img).total == len(truth)


def test_no_network_without_cache(published, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(RuntimeError, match="could not download model 'public'"):
        registry.resolve("public")


def test_checksum_mismatch(published, monkeypatch, tmp_path):
    m = registry.MODELS["public"]
    monkeypatch.setitem(
        registry.MODELS, "public", registry.Model(**{**vars(m), "sha256": "0" * 64})
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        registry.resolve("public")
    assert not any((tmp_path / "cache").iterdir())  # nothing unverified left in the cache


def test_choose_model(published):
    r = colonycount.count(make_plate()[0], model="agar")
    assert (r.model, r.model_version) == ("agar", "2.0")


def test_unknown_model(local_model):
    with pytest.raises(ValueError, match="'nope'.*available models: public"):
        colonycount.count(make_plate()[0], model="nope")


def test_unpublished_model_points_to_local_override(monkeypatch):
    monkeypatch.delenv("COLONYCOUNT_MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="not published yet.*COLONYCOUNT_MODEL_PATH"):
        registry.resolve("public")


def test_local_override(model_file, local_model, tmp_path):
    assert registry.resolve("public") == (model_file, "local")
    assert registry.resolve("public", model_path=model_file) == (model_file, "local")
    with pytest.raises(FileNotFoundError, match="nothing.onnx"):
        registry.resolve("public", model_path=tmp_path / "nothing.onnx")
