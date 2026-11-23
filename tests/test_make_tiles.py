import json
import random

import cv2
import make_tiles as mt
import numpy as np


def test_tile_boxes_clips_and_drops_mostly_outside():
    boxes = [[10, 10, 20, 20], [630, 100, 20, 20], [625, 300, 20, 20], [700, 0, 5, 5]]
    # 1: inside; 2: half inside (kept, clipped); 3: 75% inside; 4: outside
    assert mt.tile_boxes(boxes, 0, 0, 640) == [
        [10, 10, 20, 20],
        [630, 100, 10, 20],
        [625, 300, 15, 20],
    ]
    assert mt.tile_boxes([[631, 0, 20, 20]], 0, 0, 640) == []  # 45% inside


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
    assert n_tiles == 6  # default 640 px tiles, 64 px overlap

    out = tmp_path / "train"
    stats = mt.make_split(coco, tmp_path, out, 1.0, random.Random(0))
    assert stats == {"tiles": 6, "empty": 4, "boxes": 2}  # box sits in the x-overlap of 2 tiles
    tiled = json.loads((out / "_annotations.coco.json").read_text())
    assert {im["group"] for im in tiled["images"]} == {"sp01"}
    assert all((out / im["file_name"]).exists() for im in tiled["images"])
    assert [im["x"] for im in tiled["images"] if im["y"] == 0] == [0, 576, 660]

    stats = mt.make_split(coco, tmp_path, tmp_path / "t2", 0.0, random.Random(0))
    assert stats == {"tiles": 2, "empty": 0, "boxes": 2}


def test_scaled_plates_keep_boxes_on_the_scaled_image_and_photo_darkens(tmp_path):
    img = np.full((400, 400, 3), 200, np.uint8)
    img[100:140, 100:140] = 50  # a colony, box [100, 100, 40, 40]
    cv2.imwrite(str(tmp_path / "a.png"), img)
    coco = {
        "images": [{"id": 1, "file_name": "a.png", "source": "smartphone", "group": "sp01"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [100, 100, 40, 40]}],
        "categories": [{"id": 1, "name": "colony"}],
    }
    scaled, plates = mt.scaled_plates(coco, tmp_path, [0.5])
    stats = mt.make_split(scaled, tmp_path, tmp_path / "s", 0.0, random.Random(0), plates)
    assert stats["boxes"] == 1
    tiled = json.loads((tmp_path / "s" / "_annotations.coco.json").read_text())
    assert tiled["images"][0]["file_name"] == "a_s50_0_0.jpg"
    x, y, w, h = tiled["annotations"][0]["bbox"]
    tile = cv2.imread(str(tmp_path / "s" / "a_s50_0_0.jpg"))
    assert (x, y, w, h) == (50, 50, 20, 20) and tile.shape[:2] == (200, 200)
    assert tile[int(y + h / 2), int(x + w / 2)].mean() < 100 < tile[0, 0].mean()  # colony in place
    assert mt.photometric(img, random.Random(0)).mean() < img.mean()
