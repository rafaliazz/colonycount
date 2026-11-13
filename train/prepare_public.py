"""Download the public colony datasets and write leak-free COCO splits.

Sources
- smartphone: Figshare 10.6084/m9.figshare.22022540 (369 jpgs + annot_COCO.json)
- clinical: Zenodo 10.5281/zenodo.18256864, originals only (images.zip, label.zip,
  Bacterial_Image_Records.xlsx); the *_agu.zip files are flipped duplicates and skipped.

Usage: uv run --group prepare python train/prepare_public.py [--data-dir DIR] [--skip-download]

Writes <data-dir>/public/annotations/{train,val,test,adapt}.json + splits.json; image
paths are relative to the data dir. Split fractions, seed and the size of the adapt pool (taken
out of smartphone train) come from COLONYCOUNT_SPLIT_FRACTIONS, _SEED and _ADAPT_SIZE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from colonycount._env import env_int, env_path, env_tuple

FIGSHARE_API = "https://api.figshare.com/v2/articles/22022540"
ZENODO_API = "https://zenodo.org/api/records/18256864"
CLINICAL_FILES = ("Bacterial_Image_Records.xlsx", "label.zip", "images.zip")
# label.zip ships the same boxes as LabelMe (~7.6 GB, embedded images), VOC, YOLO, CSV and
# COCO; only the COCO file is kept. None = extract everything.
CLINICAL_KEEP = {"label.zip": ["label/coco_json/coco.json"]}


# --- download -------------------------------------------------------------------------


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "colonycount-prepare"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, part: Path) -> None:
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "colonycount-prepare"}
    if have:
        headers["Range"] = f"bytes={have}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        mode = "ab" if have and r.status == 206 else "wb"
        with open(part, mode) as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)


def download(url: str, dest: Path, md5: str, retries: int = 10) -> None:
    """Resumable download to `dest`, verified against `md5`; no-op if already present."""
    if dest.exists():
        return
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            _fetch(url, part)
            break
        except OSError as e:  # URLError, timeouts and resets are all OSError
            print(f"retry {attempt + 1}/{retries} {dest.name}: {e}", flush=True)
            time.sleep(10)
    else:
        raise RuntimeError(f"giving up on {dest.name}")
    got = _md5(part)
    if got != md5:
        part.unlink()
        raise RuntimeError(f"md5 mismatch for {dest.name}: {got} != {md5}")
    part.rename(dest)


def download_smartphone(root: Path) -> None:
    out = root / "smartphone"
    for f in _get_json(FIGSHARE_API)["files"]:
        if f["name"].endswith(".jpg"):
            download(f["download_url"], out / "images" / f["name"], f["computed_md5"])
        elif f["name"] == "annot_COCO.json":
            download(f["download_url"], out / f["name"], f["computed_md5"])


def download_clinical(root: Path) -> None:
    """One zip at a time: download, extract into clinical/, delete the zip."""
    out = root / "clinical"
    files = {f["key"]: f for f in _get_json(ZENODO_API)["files"]}
    for name in CLINICAL_FILES:
        f = files[name]
        md5 = f["checksum"].removeprefix("md5:")
        dest = out / name
        if not name.endswith(".zip"):
            download(f["links"]["self"], dest, md5)
            continue
        done = out / f".{name}.extracted"
        if done.exists():
            continue
        download(f["links"]["self"], dest, md5)
        with zipfile.ZipFile(dest) as z:
            # zipfile strips absolute and ".." members
            z.extractall(out, members=CLINICAL_KEEP.get(name))
        dest.unlink()
        done.touch()


# --- convert --------------------------------------------------------------------------


def convert(coco: dict, source: str, path_of, group_of) -> list[dict]:
    """Source COCO -> list of image records carrying their own boxes, source and group."""
    boxes: dict[int, list] = {}
    for a in coco["annotations"]:
        boxes.setdefault(a["image_id"], []).append(a["bbox"])
    return [
        {
            "file_name": path_of(im["file_name"]),
            "width": im["width"],
            "height": im["height"],
            "source": source,
            "group": group_of(im["file_name"]),
            "boxes": boxes.get(im["id"], []),
        }
        for im in coco["images"]
    ]


def to_coco(records: list[dict]) -> dict:
    """Image records -> COCO with fresh ids and a single "colony" category."""
    images, annotations = [], []
    for i, r in enumerate(records, 1):
        images.append({k: v for k, v in r.items() if k != "boxes"} | {"id": i})
        for b in r["boxes"]:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": b,
                    "area": b[2] * b[3],
                    "iscrowd": 0,
                }
            )
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "colony", "supercategory": "cfu"}],
    }


def _fractions() -> dict[str, float]:
    fractions = env_tuple("COLONYCOUNT_SPLIT_FRACTIONS", (0.70, 0.15, 0.15))
    return dict(zip(("train", "val", "test"), fractions, strict=True))


def _seed(seed: int | None) -> int:
    return env_int("COLONYCOUNT_SEED", 0) if seed is None else seed


def split_by_group(records: list[dict], seed: int | None = None) -> dict[str, list[dict]]:
    """Whole groups go to train/val/test; each group joins the split furthest below its
    image-count target, visiting groups in a seeded shuffle."""
    fractions = _fractions()
    by_group: dict[str, list[dict]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)
    groups = sorted(by_group)
    random.Random(_seed(seed)).shuffle(groups)
    out: dict[str, list[dict]] = {s: [] for s in fractions}
    for g in groups:
        s = max(out, key=lambda s: fractions[s] * len(records) - len(out[s]))
        out[s] += by_group[g]
    return out


def take_adapt_pool(train: list[dict], n: int | None = None, seed: int | None = None):
    """Fixed n-image pool out of `train`; returns (remaining train, pool)."""
    if n is None:
        n = env_int("COLONYCOUNT_ADAPT_SIZE", 20, lo=0)
    pool = random.Random(_seed(seed)).sample(sorted(train, key=lambda r: r["file_name"]), n)
    names = {r["file_name"] for r in pool}
    return [r for r in train if r["file_name"] not in names], pool


def build_splits(sources: dict[str, list[dict]], seed: int | None = None):
    """Per-source group split, merged; adaptation pool comes from smartphone train."""
    seed, fractions = _seed(seed), _fractions()
    splits: dict[str, list[dict]] = {s: [] for s in fractions}
    groups: dict[str, dict[str, list[str]]] = {s: {} for s in fractions}
    for source, records in sources.items():
        for s, recs in split_by_group(records, seed).items():
            groups[s][source] = sorted({r["group"] for r in recs})
            if source == "smartphone" and s == "train":
                recs, splits["adapt"] = take_adapt_pool(recs, seed=seed)
            splits[s] += recs
    manifest = {
        "seed": seed,
        "fractions": fractions,
        "groups": groups,
        "adapt": sorted(r["file_name"] for r in splits.get("adapt", [])),
    }
    return splits, manifest


def smartphone_records(root: Path) -> list[dict]:
    coco = json.loads((root / "smartphone" / "annot_COCO.json").read_text())
    return convert(
        coco,
        "smartphone",
        path_of=lambda f: f"public/smartphone/images/{f}",
        group_of=lambda f: f.split("_")[0],  # spXX_imgYY.jpg -> species spXX
    )


def clinical_isolates(xlsx: Path) -> dict[str, str]:
    """Image filename -> isolate number, from Bacterial_Image_Records.xlsx."""
    import openpyxl  # prepare dependency group only

    rows = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0].iter_rows(values_only=True)
    header = next(rows)
    fi, ii = header.index("Image filename"), header.index("Isolate number")
    return {r[fi]: r[ii] for r in rows if r[fi]}


def clinical_records(root: Path) -> list[dict]:
    coco = json.loads((root / "clinical" / "label" / "coco_json" / "coco.json").read_text())
    isolate = clinical_isolates(root / "clinical" / "Bacterial_Image_Records.xlsx")
    return convert(
        coco,
        "clinical",
        # images.zip layout: images/<species>/<species>_NN.jpg
        path_of=lambda f: f"public/clinical/images/{f.split('_')[0]}/{f}",
        group_of=lambda f: isolate[f],
    )


def main(argv: list[str] | None = None) -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, help="default: COLONYCOUNT_DATA_DIR")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args(argv)
    data_dir = (
        args.data_dir.expanduser()
        if args.data_dir
        else env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
    )
    root = data_dir / "public"
    if not args.skip_download:
        download_smartphone(root)
        download_clinical(root)

    sources = {"smartphone": smartphone_records(root), "clinical": clinical_records(root)}
    for recs in sources.values():
        missing = [r["file_name"] for r in recs if not (data_dir / r["file_name"]).exists()]
        if missing:
            raise SystemExit(f"{len(missing)} images missing, e.g. {missing[0]}")
    splits, manifest = build_splits(sources)

    ann = root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    for s, recs in splits.items():
        (ann / f"{s}.json").write_text(json.dumps(to_coco(recs)))
        counts = {src: [0, 0] for src in sources}
        for r in recs:
            counts[r["source"]][0] += 1
            counts[r["source"]][1] += len(r["boxes"])
        print(s, {src: f"{i} img / {c} colonies" for src, (i, c) in counts.items()})
    (ann / "splits.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
