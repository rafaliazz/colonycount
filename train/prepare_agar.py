"""Download the AGAR dataset (pinned Hugging Face mirror) and verify every file.

Source: https://huggingface.co/datasets/leodriesch/agar at the revision pinned below (83 files,
33.5 GB; parquet shards under data/, annotations under annotations/). Original dataset: AGAR by
NeuroSYS, CC BY-NC 2.0 (see DATA.md).

Usage: uv run --group prepare python train/prepare_agar.py [--agar-dir DIR] [--verify-only]

Files land in <agar-dir>/hf; the download is resumable and adopts files already present there
(e.g. from `hf download leodriesch/agar --local-dir <agar-dir>/hf`). Every file is then checked
against the revision's SHA-256 (LFS files) or git blob SHA-1 (small files); any missing file or
mismatch exits non-zero. <agar-dir> comes from COLONYCOUNT_AGAR_DIR, default <data dir>/agar.

TODO (task 1.1): convert to COCO and split by imaging setup. Note for that step: the data drive
may be FAT32 (about 65k files per folder, 4 GB per file), so read images straight from the
parquet shards or shard tile folders instead of writing one flat folder of images.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from dotenv import load_dotenv

from colonycount._env import env_path

REPO_ID = "leodriesch/agar"
REVISION = "9ce24ca6611a700b2c61d22946786630c65e7764"


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
    from huggingface_hub import HfApi  # prepare group only, not in CI

    tree = HfApi().list_repo_tree(
        REPO_ID, repo_type="dataset", revision=REVISION, recursive=True, expand=True
    )
    out = {}
    for f in tree:
        if not hasattr(f, "size"):  # folder
            continue
        out[f.path] = (f.size, "sha256", f.lfs.sha256) if f.lfs else (f.size, "sha1", f.blob_id)
    return out


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


def main(argv: list[str] | None = None) -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--agar-dir", type=Path, help="default: COLONYCOUNT_AGAR_DIR")
    p.add_argument("--verify-only", action="store_true", help="check files, do not download")
    args = p.parse_args(argv)
    agar_dir = (
        args.agar_dir.expanduser()
        if args.agar_dir
        else env_path("COLONYCOUNT_AGAR_DIR", None)
        or env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount") / "agar"
    )
    root = agar_dir / "hf"
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION, local_dir=root)
    problems = verify(expected_files(), root)
    if problems:
        raise SystemExit("\n".join(problems) + f"\n{len(problems)} problem(s) in {root}")
    print(f"AGAR verified in {root}")


if __name__ == "__main__":
    main()
