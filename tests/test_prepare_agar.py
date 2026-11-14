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
