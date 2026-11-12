import json
import random

import cv2
import make_tiles as mt
import numpy as np

from colonycount.detection.tiling import TILE_OVERLAP, TILE_SIZE


def test_tile_boxes_clips_and_drops_mostly_outside():
    boxes = [[10, 10, 20, 20], [630, 100, 20, 20], [625, 300, 20, 20], [700, 0, 5, 5]]
    # 1: inside; 2: half inside (kept, clipped); 3: 75% inside; 4: outside
    assert mt.tile_boxes(boxes, 0, 0) == [[10, 10, 20, 20], [630, 100, 10, 20], [625, 300, 15, 20]]
    assert mt.tile_boxes([[631, 0, 20, 20]], 0, 0) == []  # 45% inside


def test_make_split_tile_grid_metadata_and_empty_tiles(tmp_path):
    h, w = 1000, 1300  # 2 x 3 tiles on the inference grid
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((h, w, 3), np.uint8))
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": w, "height": h,
                    "source": "smartphone", "group": "sp01"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [600, 50, 20, 20]}],
        "categories": [{"id": 1, "name": "colony"}],
    }  # fmt: skip
    n_tiles = len(list(mt.tile_image(np.zeros((h, w, 3), np.uint8), [])))
    assert n_tiles == 6 and TILE_OVERLAP < TILE_SIZE

    out = tmp_path / "train"
    stats = mt.make_split(coco, tmp_path, out, 1.0, random.Random(0))
    assert stats == {"tiles": 6, "empty": 4, "boxes": 2}  # box sits in the x-overlap of 2 tiles
    tiled = json.loads((out / "_annotations.coco.json").read_text())
    assert {im["group"] for im in tiled["images"]} == {"sp01"}
    assert all((out / im["file_name"]).exists() for im in tiled["images"])
    assert [im["x"] for im in tiled["images"] if im["y"] == 0] == [0, 576, 660]

    stats = mt.make_split(coco, tmp_path, tmp_path / "t2", 0.0, random.Random(0))
    assert stats == {"tiles": 2, "empty": 0, "boxes": 2}
