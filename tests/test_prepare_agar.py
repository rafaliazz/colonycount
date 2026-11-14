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
