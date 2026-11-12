"""Checks the real prepared data; skipped unless train/prepare_public.py has been run."""

import json
import os
from pathlib import Path

import pytest

DATA = Path(os.environ.get("COLONYCOUNT_DATA_DIR", "~/data/colonycount")).expanduser()
ANN = DATA / "public" / "annotations"
pytestmark = pytest.mark.skipif(not (ANN / "splits.json").exists(), reason=f"no data in {ANN}")

PUBLISHED = {"smartphone": (369, 56865), "clinical": (950, 59221)}


def _load():
    names = ("train", "val", "test", "adapt")
    return {s: json.loads((ANN / f"{s}.json").read_text()) for s in names}


def test_totals_match_published():
    totals = {src: [0, 0] for src in PUBLISHED}
    for coco in _load().values():
        source = {im["id"]: im["source"] for im in coco["images"]}
        for im in coco["images"]:
            totals[im["source"]][0] += 1
        for a in coco["annotations"]:
            totals[source[a["image_id"]]][1] += 1
    assert {k: tuple(v) for k, v in totals.items()} == PUBLISHED


def test_no_group_in_two_splits_and_adapt_from_smartphone_train():
    splits = _load()
    groups = {
        s: {(im["source"], im["group"]) for im in splits[s]["images"]}
        for s in ("train", "val", "test")
    }
    assert not (groups["train"] & groups["val"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["val"] & groups["test"])
    adapt = splits["adapt"]["images"]
    assert len(adapt) == 20 and {im["source"] for im in adapt} == {"smartphone"}
    assert {("smartphone", im["group"]) for im in adapt} <= groups["train"]
    for coco in splits.values():
        assert all((DATA / im["file_name"]).exists() for im in coco["images"])
