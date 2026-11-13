"""bench/opencfu_runner.py: OpenCFU CSV parsing and mapping to evaluate.py's input; no data."""

import shutil
import subprocess

import cv2
import evaluate
import numpy as np
import opencfu_runner as ocfu
import pytest

CSV = """IsValid,X,Y,ROI,Colour_group,N_in_clust,Area,Radius,Hue,Saturation,Rmean
1,10,20,1,0,3,82,6,30,3,75.5
0,11,21,1,0,3,82,6,30,3,75.5
1,100.5,50,1,0,3,82,4,30,3,75.5
"""


def test_parse_keeps_valid_colonies_only():
    assert ocfu.parse(CSV).tolist() == [[10, 20, 6], [100.5, 50, 4]]
    assert ocfu.parse("").shape == (0, 3)


def test_to_plate_maps_back_and_drops_colonies_outside_the_circle():
    # resized by 0.5 from a crop starting at (1000, 2000); plate circle centred on colony 1
    prep = {"scale": 0.5, "offset": np.array([1000.0, 2000.0]), "circle": (1020.0, 2040.0, 50.0)}
    out = ocfu.to_plate(ocfu.parse(CSV), prep)
    assert out == {"xy": [(1020.0, 2040.0)], "diameter": [24.0], "conf": [1.0]}
    assert len(ocfu.to_plate(ocfu.parse(CSV), prep | {"circle": None})["xy"]) == 2


def test_scored_plate_goes_through_evaluate_metrics():
    prep = {"scale": 1.0, "offset": np.zeros(2), "circle": None}
    plate = {
        "source": "smartphone",
        "gt": 2,
        "gt_xy": np.array([[10.0, 20.0], [300.0, 300.0]]),
        "gt_r": np.array([5.0, 5.0]),
        "plate_found": False,
    } | ocfu.to_plate(ocfu.parse(CSV), prep)
    s = evaluate.summarize([plate], 1.0)
    assert (s["mae"], s["precision"], s["recall"]) == (0, 0.5, 0.5)


def test_flags_scale_radii_with_the_image():
    s = {"side": 1000, "mode": "bil -t 5", "r_min": 0.004, "r_max": 0.03}
    assert ocfu.flags(s) == "-d bil -t 5 -r 4 -R 30"
    assert ocfu.flags(s | {"mode": "auto", "side": 1500}) == "-a -r 6 -R 45"


def _image_built():
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "image", "inspect", ocfu.IMAGE], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _image_built(), reason="needs docker and the colonycount-opencfu image")
def test_opencfu_container_finds_colonies(tmp_path):
    yy, xx = np.mgrid[:400, :400]
    img = np.full((400, 400), 200.0)  # three soft dark colonies on a light, slightly noisy plate
    for x, y in [(100, 100), (300, 120), (200, 300)]:
        img -= 140 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 72)
    img += np.random.default_rng(0).normal(0, 3, img.shape)
    cv2.imwrite(str(tmp_path / "a.png"), np.clip(img, 0, 255).astype(np.uint8))
    text, seconds = ocfu.run_opencfu(tmp_path, "-d bil -t 10 -r 3 -R 30", jobs=1)["a.png"]
    assert len(ocfu.parse(text)) == 3 and seconds > 0
