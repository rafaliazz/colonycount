"""--resume handling in train_rfdetr.py; needs neither torch nor rfdetr."""

import pytest
import train_rfdetr


def test_new_run_gets_fresh_dir_and_no_resume(tmp_path):
    config = train_rfdetr.build_config(train_rfdetr.parse_args(["--data-dir", str(tmp_path)]))
    assert "resume" not in config
    assert config["output_dir"].startswith(str(tmp_path / "runs" / "rfdetr_nano"))


def test_tiles_picks_the_dataset_folder(tmp_path):
    for extra, name in (
        ([], "tiles_640"),
        (["--tiles", "tiles_640_scalephoto"], "tiles_640_scalephoto"),
    ):
        args = train_rfdetr.parse_args(["--data-dir", str(tmp_path), *extra])
        assert train_rfdetr.build_config(args)["dataset_dir"] == str(tmp_path / "public" / name)


def test_resume_continues_in_the_checkpoint_dir(tmp_path):
    run = tmp_path / "runs" / "rfdetr_nano" / "20261113-010203"
    run.mkdir(parents=True)
    ckpt = run / "last.ckpt"
    ckpt.write_bytes(b"")
    args = train_rfdetr.parse_args(["--data-dir", str(tmp_path), "--resume", str(ckpt)])
    config = train_rfdetr.build_config(args)
    assert config["resume"] == str(ckpt)
    assert config["output_dir"] == str(run)


def test_resume_missing_checkpoint_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        train_rfdetr.parse_args(["--resume", str(tmp_path / "nope.ckpt")])
