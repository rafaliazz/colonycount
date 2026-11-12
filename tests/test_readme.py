"""Docs test: run the README's `sh test` / `python test` blocks against the placeholder model."""

import re
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
from conftest import make_plate

README = Path(__file__).parents[1] / "README.md"


def test_readme_commands(tmp_path, local_model):
    blocks = re.findall(r"```(\w+) test\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert {lang for lang, _ in blocks} == {"sh", "python"}
    (tmp_path / "plates").mkdir()
    for name in ["plate.jpg", "plates/a.jpg", "plates/b.jpg"]:
        cv2.imwrite(str(tmp_path / name), make_plate()[0])
    for lang, code in blocks:
        if lang == "python":
            subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
            continue
        for line in code.splitlines():
            if cmd := shlex.split(line, comments=True):
                subprocess.run(cmd, cwd=tmp_path, check=True)
    assert (tmp_path / "counts.csv").is_file()
