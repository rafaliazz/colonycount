"""Cut the public COCO splits into TILE_SIZE tiles at native resolution for detector training.

Usage: uv run --extra train python train/make_tiles.py [--data-dir DIR] [--empty-frac 0.05]

Reads <data-dir>/public/annotations/{train,val}.json and writes the layout rfdetr expects:
<data-dir>/public/tiles_<T>/{train,valid}/ with tile jpgs + _annotations.coco.json. Tiles use the
same grid as inference (detection/tiling.py). A box is kept in a tile, clipped to it, if at least
half its area lies inside. Tiles without boxes are kept with probability --empty-frac (seeded).
Each tile image record keeps its source, group and parent image.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2

from colonycount.detection.tiling import TILE_OVERLAP, TILE_SIZE, _tiles

SEED = 0
MIN_INSIDE = 0.5
SPLITS = {"train": "train", "val": "valid"}  # ours -> rfdetr's folder names


def tile_boxes(boxes, x, y, tile=TILE_SIZE):
    """COCO xywh boxes clipped to the tile at (x, y), in tile pixels; drops < MIN_INSIDE inside."""
    out = []
    for bx, by, bw, bh in boxes:
        x1, y1 = max(bx, x), max(by, y)
        x2, y2 = min(bx + bw, x + tile), min(by + bh, y + tile)
        if x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) >= MIN_INSIDE * bw * bh:
            out.append([x1 - x, y1 - y, x2 - x1, y2 - y1])
    return out


def tile_image(img, boxes, tile=TILE_SIZE, overlap=TILE_OVERLAP):
    """Yield (x, y, crop, tile boxes) over the inference tile grid; edge crops may be smaller."""
    h, w = img.shape[:2]
    for y in _tiles(h, tile, overlap)[0]:
        for x in _tiles(w, tile, overlap)[0]:
            yield x, y, img[y : y + tile, x : x + tile], tile_boxes(boxes, x, y, tile)


def make_split(coco, data_dir: Path, out: Path, empty_frac: float, rng: random.Random) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a["bbox"])
    images, annotations, n_empty = [], [], 0
    for im in coco["images"]:
        img = cv2.imread(str(data_dir / im["file_name"]))
        if img is None:
            raise FileNotFoundError(data_dir / im["file_name"])
        stem = Path(im["file_name"]).stem
        for x, y, crop, boxes in tile_image(img, by_image[im["id"]]):
            if not boxes:
                if rng.random() >= empty_frac:
                    continue
                n_empty += 1
            tid = len(images) + 1
            name = f"{stem}_{x}_{y}.jpg"
            cv2.imwrite(str(out / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = crop.shape[:2]
            images.append(
                {"id": tid, "file_name": name, "width": w, "height": h, "source": im["source"],
                 "group": im["group"], "parent": im["file_name"], "x": x, "y": y}
            )  # fmt: skip
            for b in boxes:
                annotations.append(
                    {"id": len(annotations) + 1, "image_id": tid, "category_id": 1, "bbox": b,
                     "area": b[2] * b[3], "iscrowd": 0}
                )  # fmt: skip
    tiled = {"images": images, "annotations": annotations, "categories": coco["categories"]}
    (out / "_annotations.coco.json").write_text(json.dumps(tiled))
    return {"tiles": len(images), "empty": n_empty, "boxes": len(annotations)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path.home() / "data" / "colonycount")
    ap.add_argument("--empty-frac", type=float, default=0.05)
    args = ap.parse_args()
    root = args.data_dir / "public" / f"tiles_{TILE_SIZE}"
    rng = random.Random(SEED)
    for split, folder in SPLITS.items():
        coco = json.loads((args.data_dir / "public" / "annotations" / f"{split}.json").read_text())
        stats = make_split(coco, args.data_dir, root / folder, args.empty_frac, rng)
        print(split, stats, flush=True)


if __name__ == "__main__":
    main()
