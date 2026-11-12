import prepare_public as pp


def _source_coco(n_groups=10, per_group=5):
    images, annotations = [], []
    for g in range(n_groups):
        for k in range(per_group):
            iid = len(images) + 1
            name = f"sp{g:02d}_img{k:02d}.jpg"
            images.append({"id": iid, "file_name": name, "width": 9, "height": 9})
            for _ in range(g + 1):
                aid = len(annotations) + 1
                annotations.append({"id": aid, "image_id": iid, "bbox": [1, 2, 3, 4]})
    return {"images": images, "annotations": annotations}


def _records(source="smartphone", **kw):
    return pp.convert(_source_coco(**kw), source, lambda f: f"x/{f}", lambda f: f.split("_")[0])


def test_convert_keeps_boxes_and_tags_source_and_group():
    recs = _records()
    assert len(recs) == 50 and sum(len(r["boxes"]) for r in recs) == 5 * sum(range(1, 11))
    assert recs[0]["file_name"] == "x/sp00_img00.jpg"
    assert {r["source"] for r in recs} == {"smartphone"}
    assert recs[-1]["group"] == "sp09"

    coco = pp.to_coco(recs)
    assert len({im["id"] for im in coco["images"]}) == 50
    assert len({a["id"] for a in coco["annotations"]}) == len(coco["annotations"]) == 275
    assert coco["annotations"][0]["area"] == 12
    assert {a["category_id"] for a in coco["annotations"]} == {1}
    assert "boxes" not in coco["images"][0] and coco["images"][0]["source"] == "smartphone"


def test_split_by_group_is_leak_free_complete_and_seeded():
    recs = _records(n_groups=20)
    splits = pp.split_by_group(recs)
    seen = [{r["group"] for r in s} for s in splits.values()]
    assert all(not (a & b) for i, a in enumerate(seen) for b in seen[i + 1 :])
    assert sum(len(s) for s in splits.values()) == len(recs)
    assert [len(s) for s in splits.values()] == [70, 15, 15]
    assert pp.split_by_group(recs) == splits
    assert pp.split_by_group(recs, seed=1) != splits


def test_build_splits_takes_adapt_pool_from_smartphone_train():
    sources = {"smartphone": _records(n_groups=20), "clinical": _records("clinical", n_groups=20)}
    splits, manifest = pp.build_splits(sources)
    adapt = splits["adapt"]
    assert len(adapt) == 20 and {r["source"] for r in adapt} == {"smartphone"}
    assert {r["group"] for r in adapt} <= set(manifest["groups"]["train"]["smartphone"])
    names = {r["file_name"] + r["source"] for s in splits.values() for r in s}
    assert len(names) == 200  # nothing dropped or duplicated
    assert manifest["adapt"] == sorted(r["file_name"] for r in adapt)
    assert pp.build_splits(sources) == (splits, manifest)
