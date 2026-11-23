"""Cut the public COCO splits into square tiles at native resolution for detector training.

Usage: uv run --extra train python train/make_tiles.py [--data-dir DIR] [--empty-frac 0.05]

Reads <data-dir>/public/annotations/{train,val}.json and writes the layout rfdetr expects:
<data-dir>/public/tiles_<T>/{train,valid}/ with tile jpgs + _annotations.coco.json, T =
COLONYCOUNT_TRAIN_TILE_SIZE. Tiles use the same grid as inference (detection/tiling.py). A box is
kept in a tile, clipped to it, if at least COLONYCOUNT_TILE_MIN_INSIDE of its area lies inside.
Tiles without boxes are kept with probability --empty-frac (seeded by COLONYCOUNT_SEED).
Each tile image record keeps its source, group and parent image.

Robustness recipes (train only): --scale S [S ...] also tiles every train plate downscaled by S
(colonies get smaller), --photo adds a copy of every original train tile with its contrast cut
towards the tile mean (alpha 0.5-0.9) and darkened (gamma 1.2-2.0), drawn per tile from the seed.
The result is <data-dir>/public/tiles_<T>_<scale|photo|scalephoto>/{train,valid}: the original
tiles (hard-linked, so tiles_<T> must exist) plus the extra ones, and the original val tiles.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from colonycount._env import env_float, env_int, env_path
from colonycount.detection.tiling import _tiles

SPLITS = {"train": "train", "val": "valid"}  # ours -> rfdetr's folder names


def tile_boxes(boxes, x, y, tile):
    """COCO xywh boxes clipped to the tile at (x, y), in tile pixels; drops mostly-outside ones."""
    min_inside = env_float("COLONYCOUNT_TILE_MIN_INSIDE", 0.5, 0, 1)
    out = []
    for bx, by, bw, bh in boxes:
        x1, y1 = max(bx, x), max(by, y)
        x2, y2 = min(bx + bw, x + tile), min(by + bh, y + tile)
        if x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) >= min_inside * bw * bh:
            out.append([x1 - x, y1 - y, x2 - x1, y2 - y1])
    return out


def tile_image(img, boxes):
    """Yield (x, y, crop, tile boxes) over the inference tile grid; edge crops may be smaller."""
    tile = env_int("COLONYCOUNT_TRAIN_TILE_SIZE", 640, lo=1)
    overlap = env_int("COLONYCOUNT_TILE_OVERLAP", 64, lo=0)
    h, w = img.shape[:2]
    for y in _tiles(h, tile, overlap)[0]:
        for x in _tiles(w, tile, overlap)[0]:
            yield x, y, img[y : y + tile, x : x + tile], tile_boxes(boxes, x, y, tile)


def read_files(coco, data_dir: Path):
    """Yield (image record, BGR image) for every image of the COCO, read from `data_dir`."""
    for im in coco["images"]:
        img = cv2.imread(str(data_dir / im["file_name"]))
        if img is None:
            raise FileNotFoundError(data_dir / im["file_name"])
        yield im, img


def make_split(
    coco, data_dir: Path, out: Path, empty_frac: float, rng: random.Random, plates=None
) -> dict:
    """Tile `plates` (default: the COCO's images read from `data_dir`; any iterator of
    (image record, BGR image) works). An image record with a "shard" has its tiles written
    into that sub-folder of `out`, so big sets stay under FAT32's ~65k files per folder."""
    out.mkdir(parents=True, exist_ok=True)
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a["bbox"])
    images, annotations, n_empty = [], [], 0
    for im, img in plates or read_files(coco, data_dir):
        stem = Path(im["file_name"]).stem
        shard = im.get("shard", "")
        (out / shard).mkdir(exist_ok=True)
        for x, y, crop, boxes in tile_image(img, by_image[im["id"]]):
            if not boxes:
                if rng.random() >= empty_frac:
                    continue
                n_empty += 1
            tid = len(images) + 1
            name = f"{stem}_{x}_{y}.jpg" if not shard else f"{shard}/{stem}_{x}_{y}.jpg"
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
    if "info" in coco:
        tiled["info"] = coco["info"]
    (out / "_annotations.coco.json").write_text(json.dumps(tiled))
    return {"tiles": len(images), "empty": n_empty, "boxes": len(annotations)}


def scaled_plates(coco, data_dir: Path, scales):
    """(COCO, plates) of every plate downscaled by each factor, boxes scaled with it; the scaled
    images get ids and file names (``_s50`` for 0.5) of their own."""
    images, annotations = [], []
    for k, s in enumerate(scales):
        for im in coco["images"]:
            stem, ext = os.path.splitext(im["file_name"])
            name = f"{stem}_s{round(s * 100)}{ext}"
            new_id = im["id"] + (k + 1) * 10**7
            images.append(
                {**im, "id": new_id, "file_name": name, "src": im["file_name"], "scale": s}
            )
        annotations += [
            {**a, "image_id": a["image_id"] + (k + 1) * 10**7, "bbox": [v * s for v in a["bbox"]]}
            for a in coco["annotations"]
        ]

    def plates():
        for im in images:
            img = cv2.imread(str(data_dir / im["src"]))
            if img is None:
                raise FileNotFoundError(data_dir / im["src"])
            yield (
                im,
                cv2.resize(img, None, fx=im["scale"], fy=im["scale"], interpolation=cv2.INTER_AREA),
            )

    return {**coco, "images": images, "annotations": annotations}, plates()


def photometric(img, rng: random.Random):
    """Contrast reduced toward the tile mean (alpha 0.5-0.9), then darkened (gamma 1.2-2.0)."""
    alpha, gamma = rng.uniform(0.5, 0.9), rng.uniform(1.2, 2.0)
    x = img.astype(np.float32)
    x = np.clip(x.mean() + alpha * (x - x.mean()), 0, 255)
    return (255 * (x / 255) ** gamma).round().astype(np.uint8)


def photo_copies(coco, src: Path, out: Path, rng: random.Random) -> dict:
    """COCO of a photometric() copy, written to `out`, of each tile of `coco` read from `src`."""
    images, annotations = [], []
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)
    for im in coco["images"]:
        stem, ext = os.path.splitext(im["file_name"])
        name = f"{stem}_photo{ext}"
        cv2.imwrite(str(out / name), photometric(cv2.imread(str(src / im["file_name"])), rng),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])  # fmt: skip
        images.append({**im, "id": len(images) + 1, "file_name": name})
        annotations += [{**a, "image_id": len(images)} for a in by_image[im["id"]]]
    return {"images": images, "annotations": annotations}


def link_tiles(src: Path, dst: Path) -> None:
    """Hard-link the tile jpgs of `src` into `dst` (same volume); the COCO json is copied."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".jpg" and not (dst / f.name).exists():
            os.link(f, dst / f.name)
    shutil.copy2(src / "_annotations.coco.json", dst)


def add(base: dict, extra: dict) -> None:
    """Append `extra`'s images and annotations to `base`, renumbering their ids."""
    n_img, n_ann = len(base["images"]), len(base["annotations"])
    new = {im["id"]: n_img + i + 1 for i, im in enumerate(extra["images"])}
    base["images"] += [{**im, "id": new[im["id"]]} for im in extra["images"]]
    base["annotations"] += [
        {**a, "id": n_ann + i + 1, "image_id": new[a["image_id"]]}
        for i, a in enumerate(extra["annotations"])
    ]


def main() -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, help="default: COLONYCOUNT_DATA_DIR")
    ap.add_argument("--empty-frac", type=float, default=0.05)
    ap.add_argument("--scale", type=float, nargs="+", help="also tile train plates downscaled by S")
    ap.add_argument("--photo", action="store_true", help="add contrast/exposure copies of train")
    args = ap.parse_args()
    data_dir = args.data_dir or env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
    tile = env_int("COLONYCOUNT_TRAIN_TILE_SIZE", 640, lo=1)
    root = data_dir / "public" / f"tiles_{tile}"
    rng = random.Random(env_int("COLONYCOUNT_SEED", 0))
    if args.scale or args.photo:
        base = root
        recipe = ("scale" if args.scale else "") + ("photo" if args.photo else "")
        root = data_dir / "public" / f"tiles_{tile}_{recipe}"
        merged = json.loads((base / "train" / "_annotations.coco.json").read_text())
        link_tiles(base / "train", root / "train")
        link_tiles(base / "valid", root / "valid")
        if args.scale:
            coco = json.loads((data_dir / "public" / "annotations" / "train.json").read_text())
            coco, plates = scaled_plates(coco, data_dir, args.scale)
            print(
                "scaled", make_split(coco, data_dir, root / "train", args.empty_frac, rng, plates)
            )
            add(merged, json.loads((root / "train" / "_annotations.coco.json").read_text()))
        if args.photo:
            orig = json.loads((base / "train" / "_annotations.coco.json").read_text())
            add(merged, photo_copies(orig, base / "train", root / "train", rng))
        (root / "train" / "_annotations.coco.json").write_text(json.dumps(merged))
        print("train", len(merged["images"]), "tiles", len(merged["annotations"]), "boxes")
        return
    for split, folder in SPLITS.items():
        coco = json.loads((data_dir / "public" / "annotations" / f"{split}.json").read_text())
        stats = make_split(coco, data_dir, root / folder, args.empty_frac, rng)
        print(split, stats, flush=True)


if __name__ == "__main__":
    main()
