"""Configuration from the environment / .env, .env.dist completeness and the model contract."""

import ast
import csv
from pathlib import Path

import cv2
import onnx
import pytest
from conftest import make_plate
from dotenv import dotenv_values
from onnx import TensorProto, helper

import colonycount
from colonycount import _env
from colonycount.cli import main
from colonycount.detection.runner import load_session
from colonycount.utils.cfu import range_flag

ROOT = Path(__file__).parents[1]
CODE = [
    p
    for d in ("src", "train", "bench")
    for p in sorted((ROOT / d).rglob("*.py"))
    if "third_party" not in p.parts  # vendored as-is (train/third_party/yolox)
]


def test_precedence_flag_env_dotenv_default(tmp_path, monkeypatch, local_model):
    monkeypatch.chdir(tmp_path)
    cv2.imwrite("plate.jpg", make_plate()[0])
    Path(".env").write_text("COLONYCOUNT_THRESHOLD=0.7\nCOLONYCOUNT_COUNTABLE_RANGE=1,2\n")
    monkeypatch.setenv("COLONYCOUNT_COUNTABLE_RANGE", "50,300")
    img = make_plate()[0]
    assert colonycount.count(img).threshold == 0.5  # the library never reads .env itself

    def run(*flags):
        assert main(["plate.jpg", "-o", "out.csv", *flags]) == 0
        with open("out.csv", newline="") as f:
            return next(csv.DictReader(f))

    row = run()
    assert row["threshold"] == "0.7"  # .env beats the default
    assert row["range_flag"] == "TFTC"  # real env (50,300) beats .env (1,2)
    assert row["model"] == "rfdetr-nano"  # default
    assert run("--threshold", "0.6")["threshold"] == "0.6"  # flag beats .env
    assert colonycount.count(img, threshold=0.9).threshold == 0.9  # argument beats env


@pytest.mark.parametrize(
    "name, value, read",
    [
        ("COLONYCOUNT_THRESHOLD", "high", lambda: colonycount.count(make_plate()[0])),
        ("COLONYCOUNT_THRESHOLD", "1.5", lambda: colonycount.count(make_plate()[0])),
        ("COLONYCOUNT_NMS_IOU", "-1", lambda: colonycount.count(make_plate()[0])),
        ("COLONYCOUNT_PLATE_SEARCH_SIZE", "0", lambda: colonycount.count(make_plate()[0])),
        ("COLONYCOUNT_TILE_OVERLAP", "6.5", lambda: colonycount.count(make_plate()[0])),
        ("COLONYCOUNT_COUNTABLE_RANGE", "30", lambda: range_flag(10)),
    ],
)
def test_invalid_value_names_the_variable(name, value, read, monkeypatch, local_model):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"{name}='{value}' is invalid"):
        read()


def test_invalid_value_is_a_cli_usage_error(monkeypatch, capsys):
    monkeypatch.setenv("COLONYCOUNT_THRESHOLD", "high")
    with pytest.raises(SystemExit) as e:
        main(["plate.jpg"])
    assert e.value.code == 2 and "COLONYCOUNT_THRESHOLD" in capsys.readouterr().err


def _model(
    path, name="images", dtype=TensorProto.FLOAT, shape=(1, 3, 8, 8), outs=("boxes", "scores")
):
    nodes = [helper.make_node("Identity", [name], [o]) for o in outs]
    graph = helper.make_graph(
        nodes,
        "m",
        [helper.make_tensor_value_info(name, dtype, list(shape))],
        [helper.make_tensor_value_info(o, dtype, list(shape)) for o in outs],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"name": "input"}, "expected one input 'images'"),
        ({"dtype": TensorProto.DOUBLE}, "must be float32"),
        ({"shape": (1, 3, "H", "W")}, r"\[1, 3, T, T\] with a fixed T"),
        ({"shape": (1, 3, 8, 16)}, r"\[1, 3, T, T\] with a fixed T"),
        ({"outs": ("boxes",)}, r"missing output\(s\) \['scores'\]"),
    ],
)
def test_model_contract_is_checked_on_load(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        load_session(_model(tmp_path / "m.onnx", **kw))


def _env_reads():
    """(helper, variable, default) for every COLONYCOUNT_* read in src/, train/, bench/."""
    reads = []
    for py in CODE:
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "").startswith("env_"):
                name, default = (ast.literal_eval(a) for a in node.args[:2])
                reads.append((getattr(_env, node.func.id), name, default))
    return reads


def test_env_dist_documents_every_variable_with_its_default(monkeypatch):
    dist = dotenv_values(ROOT / ".env.dist")
    reads = _env_reads()
    assert {name for _, name, _ in reads} == {k for k in dist if k.startswith("COLONYCOUNT_")}
    for read, name, default in reads:
        assert (dist[name] == "") == (default is None), name
        monkeypatch.delenv(name, raising=False)
        expected = read(name, default)
        monkeypatch.setenv(name, dist[name])
        assert read(name, default) == expected, f"{name}: .env.dist and code defaults differ"


def test_no_uppercase_names_imported_across_modules():
    local = {"colonycount", *(p.stem for p in CODE if p.parent.name in ("train", "bench"))}
    bad = [
        f"{py.relative_to(ROOT)}: {a.name}"
        for py in CODE
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
        and (node.level or (node.module or "").split(".")[0] in local)
        for a in node.names
        if a.name.isupper()
    ]
    assert not bad
