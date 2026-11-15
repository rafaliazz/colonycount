import hashlib

import prepare_agar as pa


def test_verify_flags_missing_file_and_checksum_mismatch(tmp_path):
    (tmp_path / "good.bin").write_bytes(b"abc")
    (tmp_path / "bad.bin").write_bytes(b"abd")
    sha = hashlib.sha256(b"abc").hexdigest()
    expected = {
        "good.bin": (3, "sha256", sha),
        "bad.bin": (3, "sha256", sha),
        "gone.bin": (3, "sha256", sha),
    }
    assert pa.verify(expected, tmp_path) == ["checksum mismatch: bad.bin", "missing: gone.bin"]


def test_expected_files_follows_next_link_and_keeps_files_only(monkeypatch):
    import io
    import json
    from email.message import Message

    first = f"http://x/api/datasets/{pa.REPO_ID}/tree/{pa.REVISION}?recursive=true"
    pages = {
        first: (
            [
                {"type": "directory", "path": "d"},
                {"type": "file", "path": "a", "size": 1, "oid": "s1"},
            ],
            '<http://x/p2>; rel="next"',
        ),
        "http://x/p2": (
            [{"type": "file", "path": "d/b", "size": 2, "oid": "s2", "lfs": {"oid": "h2"}}],
            "",
        ),
    }

    def fake_urlopen(req, timeout):
        body, link = pages[req.full_url]
        r = io.BytesIO(json.dumps(body).encode())
        r.headers = Message()
        r.headers["Link"] = link
        return r

    monkeypatch.setattr(pa.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(pa, "API", "http://x")
    assert pa.expected_files() == {"a": (1, "sha1", "s1"), "d/b": (2, "sha256", "h2")}


def test_split_keeps_identical_images_in_one_split_and_is_seeded():
    # ids 2p and 2p+1 are the same file (same group); 40 plates, 80 images
    records = [
        {"image_id": i, "source": "dark", "group": f"g{i // 2}", "boxes": [], "file_name": f"{i}"}
        for i in range(80)
    ]

    def split(seed):
        out, _ = pa.build_splits({"dark": records}, seed=seed)
        return {s: sorted(r["image_id"] for r in rs) for s, rs in out.items()}

    a = split(0)
    assert a == split(0) and a != split(1)
    assert sum(map(len, a.values())) == 80
    for ids in a.values():
        assert all((i ^ 1) in ids for i in ids)  # the twin is in the same split
