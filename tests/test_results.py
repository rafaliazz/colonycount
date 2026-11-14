"""results/benchmark: outputs parse, hold no absolute paths, and comparison.md is up to date."""

import json
from pathlib import Path

import publish_results
import pytest

RESULTS = Path(__file__).parent.parent / "results" / "benchmark"
FILES = sorted(p for p in RESULTS.rglob("*") if p.is_file())


def test_results_exist_and_parse():
    assert any(p.suffix == ".json" for p in FILES)
    for p in FILES:
        if p.suffix == ".json":
            json.loads(p.read_text())
        elif p.suffix == ".jsonl":
            [json.loads(x) for x in p.read_text().splitlines()]


def test_no_absolute_paths():
    for p in FILES:
        assert not publish_results.ABSOLUTE.search(p.read_text()), p


def test_comparison_is_regenerated():
    for d in {p.parent for p in FILES}:
        assert (d / "comparison.md").read_text() == publish_results.comparison(d), d


def test_relative_rewrites_paths_under_root_and_rejects_others(tmp_path):
    inside = str(tmp_path / "public" / "a.jpg")
    assert publish_results.relative({"f": [inside, "x"]}, tmp_path.resolve()) == {
        "f": ["public/a.jpg", "x"]
    }
    with pytest.raises(ValueError):
        publish_results.relative(r"C:\elsewhere\x.jpg", tmp_path.resolve())


def test_readmes_are_regenerated_and_match_the_json():
    root = Path(__file__).parent.parent
    folder = RESULTS / "val"
    full, short = (root / "results" / "README.md"), (root / "README.md")
    block = publish_results.report(folder)
    assert publish_results.splice(full.read_text(), block) == full.read_text()
    block = publish_results.readme_summary(folder, "results/README.md")
    assert publish_results.splice(short.read_text(), block) == short.read_text()
    mdape = json.loads((folder / "rfdetr-nano.json").read_text())["by_source"]["overall"]
    assert f"| **RF-DETR-nano** (ships) | {mdape['mdape_30_300']:.2f}% |" in short.read_text()
