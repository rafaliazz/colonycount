"""Download the AGAR dataset (pinned Hugging Face mirror) and verify every file.

Source: https://huggingface.co/datasets/leodriesch/agar at the revision pinned below (83 files,
33.5 GB; parquet shards under data/, annotations under annotations/). Original dataset: AGAR by
NeuroSYS, CC BY-NC 2.0 (see DATA.md).

Usage: uv run --group prepare python train/prepare_agar.py [--agar-dir DIR] [--skip-download]

Files land in <agar-dir>/hf; the download is resumable and adopts files already present there
(e.g. from `hf download leodriesch/agar --local-dir <agar-dir>/hf`); plain HTTP via the standard
library, no huggingface_hub (its Xet transfer was ~200x slower here). Every file is then checked
against the revision's SHA-256 (LFS files) or git blob SHA-1 (small files); any missing file or
mismatch exits non-zero. <agar-dir> comes from COLONYCOUNT_AGAR_DIR, default <data dir>/agar.

Then converts the mirror into leak-free COCO splits and training tiles (<agar-dir>, not in git):
- annotations/{train,val,test}.json: one record per plate, same format as prepare_public.py;
  annotations/splits.json: seed, resize factors, groups and the uncountable plates (no boxes, in
  no split).
- plates/{val,test}/<shard>/<id>.jpg: the val/test plates at the resized scale (bench input).
- tiles_<T>/{train,valid}/<shard>/: training tiles from make_tiles.py (T = train tile size).
Plates are decoded straight from the parquet bytes. Colony boxes only (Defect/Contamination are
dropped). Plates are resized per imaging setup so the median colony box is about TARGET_SIDE px,
like the public mix; the factors are in the json `info`. The split is by plate group (a plate
with the same bytes stays in one split), stratified by setup, with COLONYCOUNT_SEED and
COLONYCOUNT_SPLIT_FRACTIONS. The data drive may be FAT32 (about 65k files per folder, 4 GB per
file): tiles and plates go into shard folders. --skip-download skips the download and the
checksum pass (no network).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from make_tiles import make_split
from prepare_public import build_splits, download, to_coco

from colonycount._env import env_int, env_path

API = "https://huggingface.co"
REPO_ID = "leodriesch/agar"
REVISION = "9ce24ca6611a700b2c61d22946786630c65e7764"
BACKGROUNDS = ("bright", "dark", "vague", "lower_resolution")  # parquet `background` class order
EMPTY, COUNTABLE = 0, 1  # parquet `image_type`; 2 = uncountable
N_COLONY_CLASSES = 5  # `objects.category` 0-4 are the microorganisms; 5-6 Defect, Contamination
TARGET_SIDE = 64  # px; median colony box side of the public mix is 55 (smartphone), 70 (clinical)


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    if algo == "sha1":  # git blob id: sha1 over "blob <size>\0" + content
        h.update(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def expected_files() -> dict[str, tuple[int, str, str]]:
    """{path: (size, algo, hex digest)} for every file of the pinned revision."""
    url = f"{API}/api/datasets/{REPO_ID}/tree/{REVISION}?recursive=true"
    out = {}
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "colonycount-prepare"})
        with urllib.request.urlopen(req, timeout=60) as r:
            entries = json.load(r)
            link = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        url = link and link.group(1)
        for f in entries:
            if f["type"] == "file":
                lfs = f.get("lfs")
                out[f["path"]] = (
                    (f["size"], "sha256", lfs["oid"]) if lfs else (f["size"], "sha1", f["oid"])
                )
    return out


def fetch_all(expected: dict[str, tuple[int, str, str]], root: Path, workers: int = 4) -> None:
    """Download every file not already present (right size) into `root`, a few at a time."""

    def one(item):
        rel, (size, algo, digest) = item
        if (root / rel).is_file():
            return
        url = f"{API}/datasets/{REPO_ID}/resolve/{REVISION}/{urllib.parse.quote(rel)}"
        download(url, root / rel, digest, digest=lambda p: _digest(p, algo))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, expected.items()))


def verify(expected: dict[str, tuple[int, str, str]], root: Path) -> list[str]:
    """Problems found under `root` (empty list = every file present and intact)."""
    problems = []
    for rel, (size, algo, digest) in sorted(expected.items()):
        p = root / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif p.stat().st_size != size:
            problems.append(f"size mismatch: {rel} ({p.stat().st_size} != {size})")
        elif _digest(p, algo) != digest:
            problems.append(f"checksum mismatch: {rel}")
        else:
            print(f"ok {rel}", flush=True)
    return problems


# --- convert --------------------------------------------------------------------------


def read_records(root: Path) -> list[dict]:
    """One record per image from the parquet shards. The group is the hash of the JPEG bytes:
    identical files (the release has 6 pairs, 4 of them across its own splits) are one plate."""
    import pyarrow.parquet as pq  # prepare dependency group only

    out = []
    for f in sorted((root / "hf" / "data").glob("*/*.parquet")):
        batches = pq.ParquetFile(f).iter_batches(16)  # whole shards are ~0.5 GB of jpg bytes
        for r in (row for b in batches for row in b.to_pylist()):
            o = r["objects"]
            boxes = [
                b for b, c in zip(o["bbox"], o["category"], strict=True) if c < N_COLONY_CLASSES
            ]
            out.append(
                {
                    "image_id": r["image_id"],
                    "source": BACKGROUNDS[r["background"]],
                    "type": r["image_type"],
                    "width": r["width"],
                    "height": r["height"],
                    "boxes": boxes,
                    "group": hashlib.sha1(r["image"]["bytes"]).hexdigest()[:12],
                    "parquet": f.relative_to(root).as_posix(),
                }
            )
    return out


def resize_factors(records: list[dict]) -> dict[str, float]:
    """Per imaging setup: scale (never up) bringing the median colony box side to TARGET_SIDE."""
    sides: dict[str, list] = {}
    for r in records:
        sides.setdefault(r["source"], []).extend((w * h) ** 0.5 for _, _, w, h in r["boxes"])
    return {s: round(min(1.0, TARGET_SIDE / float(np.median(v))), 2) for s, v in sides.items()}


def plate_records(records: list[dict], factors: dict[str, float]) -> list[dict]:
    """Countable and empty plates at the resized scale, with shard folder."""
    out = []
    for r in records:
        if r["type"] not in (EMPTY, COUNTABLE):
            continue
        f = factors[r["source"]]
        out.append(
            r
            | {
                "width": round(r["width"] * f),
                "height": round(r["height"] * f),
                "boxes": [[round(v * f, 1) for v in b] for b in r["boxes"]],
                "shard": f"{r['image_id'] // 500:03d}",
            }
        )
    return out


def stream(root: Path, coco: dict, save_to: Path | None = None):
    """Yield (image record, resized BGR plate) for the COCO's images in parquet order, decoded
    from the parquet bytes; with `save_to`, also write each plate there as a jpg."""
    import pyarrow.parquet as pq  # prepare dependency group only

    by_id = {im["image_id"]: im for im in coco["images"]}
    for shard in sorted({im["parquet"] for im in coco["images"]}):
        for b in pq.ParquetFile(root / shard).iter_batches(8, columns=["image", "image_id"]):
            for img, i in zip(
                b.column("image").to_pylist(), b.column("image_id").to_pylist(), strict=True
            ):
                if i not in by_id:
                    continue
                im = by_id[i]
                bgr = cv2.imdecode(np.frombuffer(img["bytes"], np.uint8), cv2.IMREAD_COLOR)
                bgr = cv2.resize(bgr, (im["width"], im["height"]), interpolation=cv2.INTER_AREA)
                if save_to:
                    dest = save_to / im["file_name"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                yield im, bgr


def convert(agar_dir: Path) -> None:
    records = read_records(agar_dir)
    factors = resize_factors([r for r in records if r["type"] == COUNTABLE])
    plates = plate_records(records, factors)
    for r in plates:  # file_name: where the resized plate is written (val/test only)
        r["file_name"] = f"plates/{{split}}/{r['shard']}/{r['image_id']}.jpg"
    splits, manifest = build_splits(
        {s: [r for r in plates if r["source"] == s] for s in BACKGROUNDS}
    )
    manifest |= {
        "resize": factors,
        "uncountable": {
            s: [r["image_id"] for r in records if r["source"] == s and r["type"] not in (0, 1)]
            for s in BACKGROUNDS
        },
    }
    info = {"description": "AGAR, colony boxes only", "resize": factors, "seed": manifest["seed"]}
    ann = agar_dir / "annotations"
    ann.mkdir(exist_ok=True)
    cocos = {}
    for split, recs in splits.items():
        for r in recs:
            r["file_name"] = r["file_name"].format(split=split)
        cocos[split] = to_coco(recs) | {"info": info}
        (ann / f"{split}.json").write_text(json.dumps(cocos[split]))
        print(split, len(recs), "plates", sum(len(r["boxes"]) for r in recs), "colonies")
    (ann / "splits.json").write_text(json.dumps(manifest, indent=1))

    tile = env_int("COLONYCOUNT_TRAIN_TILE_SIZE", 640, lo=1)
    rng = random.Random(env_int("COLONYCOUNT_SEED", 0))
    for split, folder, save_to in (("train", "train", None), ("val", "valid", agar_dir)):
        plates_iter = stream(agar_dir, cocos[split], save_to)
        out = agar_dir / f"tiles_{tile}" / folder
        stats = make_split(cocos[split], agar_dir, out, 0.05, rng, plates=plates_iter)
        print(split, stats, flush=True)
    deque(stream(agar_dir, cocos["test"], agar_dir), maxlen=0)  # writes the test plates


def main(argv: list[str] | None = None) -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--agar-dir", type=Path, help="default: COLONYCOUNT_AGAR_DIR")
    p.add_argument("--skip-download", action="store_true", help="no download or checksums")
    args = p.parse_args(argv)
    agar_dir = (
        args.agar_dir.expanduser()
        if args.agar_dir
        else env_path("COLONYCOUNT_AGAR_DIR", None)
        or env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount") / "agar"
    )
    root = agar_dir / "hf"
    if not args.skip_download:
        expected = expected_files()
        fetch_all(expected, root)
        problems = verify(expected, root)
        if problems:
            raise SystemExit("\n".join(problems) + f"\n{len(problems)} problem(s) in {root}")
        print(f"AGAR verified in {root}")
    convert(agar_dir)


if __name__ == "__main__":
    main()
