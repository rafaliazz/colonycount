"""counting-cli spec scenarios."""

import csv
import json

import cv2
import pytest
from conftest import make_plate

import colonycount
from colonycount import cli
from colonycount.cli import main


@pytest.fixture
def plates(tmp_path, local_model):
    d = tmp_path / "plates"
    d.mkdir()
    for i in range(3):
        img, _ = make_plate(seed=i)
        cv2.imwrite(str(d / f"plate{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return d


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == cli.COLUMNS
    return {r["file"].replace("\\", "/").rsplit("/", 1)[-1]: r for r in rows}


def test_folder_of_plates(plates, tmp_path):
    out = tmp_path / "counts.csv"
    args = [str(plates / "*.jpg"), "--dilution", "1e-4", "--volume", "0.1", "-o", str(out)]
    assert main(args) == 0
    rows = read(out)
    assert sorted(rows) == ["plate0.jpg", "plate1.jpg", "plate2.jpg"]
    for r in rows.values():
        assert r["count"] == "40" and r["plate_found"] == "True" and r["range_flag"] == "OK"
        assert float(r["cfu_per_ml"]) == pytest.approx(40 / 1e-5)
        assert (r["model"], r["model_version"]) == ("rfdetr-nano", "local")
        assert r["threshold"] == "0.5" and r["package_version"] == colonycount.__version__
        assert r["error"] == ""


def test_one_corrupt_file(plates, tmp_path):
    (plates / "broken.jpg").write_bytes(b"\xff\xd8 truncated")
    out = tmp_path / "counts.csv"
    assert main([str(plates / "*.jpg"), str(tmp_path / "nomatch*.png"), "-o", str(out)]) == 1
    rows = read(out)
    assert all(rows[f"plate{i}.jpg"]["count"] == "40" for i in range(3))
    assert "broken.jpg" in rows["broken.jpg"]["error"] and rows["broken.jpg"]["count"] == ""
    assert rows["nomatch*.png"]["error"]
    assert rows["plate0.jpg"]["cfu_per_ml"] == ""  # no dilution/volume given


def test_export_detections(plates, tmp_path):
    out = tmp_path / "out"
    assert (
        main([str(plates / "*.jpg"), "--detections", str(out), "-o", str(tmp_path / "c.csv")]) == 0
    )
    assert sorted(p.name for p in out.iterdir()) == ["plate0.json", "plate1.json", "plate2.json"]
    dets = json.loads((out / "plate0.json").read_text())["detections"]
    assert len(dets) == 40 and set(dets[0]) == {"x", "y", "diameter", "confidence"}


def test_range_and_threshold_options(plates, tmp_path):
    out = tmp_path / "c.csv"
    args = [str(plates / "plate0.jpg"), "--range", "50", "300", "--threshold", "0.7"]
    assert main([*args, "-o", str(out)]) == 0
    r = read(out)["plate0.jpg"]
    assert r["range_flag"] == "TFTC" and r["threshold"] == "0.7" and int(r["count"]) < 40


def test_list_models(capsys):
    assert main(["--list-models"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("name\tversion\tsize MB\tparams M\tCPU s/plate\tthreshold\tlicense")
    assert "rfdetr-nano (default)\t0.1.0\t109.2\t27.2\t8.7\t0.4\tCC-BY-4.0" in out
    assert "yolox-tiny\t0.1.0\t20.3\t5.0\t1.3\t0.5\tCC-BY-4.0" in out


@pytest.mark.parametrize(
    "args",
    [
        ["--dilution", "0", "--volume", "0.1"],
        ["--dilution", "1e-4"],
        ["--model", "nope"],
        ["--range", "300", "30"],
    ],
)
def test_parameter_errors(plates, args):
    with pytest.raises(SystemExit) as e:
        main([str(plates / "*.jpg"), *args])
    assert e.value.code == 2
