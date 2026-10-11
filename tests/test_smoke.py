"""Smoke tests for the Phase-0 skeleton."""

import numpy as np
import pytest

from aegis.cli import _coerce_source, build_parser
from aegis.config import load_site_config
from aegis.video import VideoSource


def test_coerce_source_index():
    assert _coerce_source("0") == 0
    assert _coerce_source("data/clip.mp4") == "data/clip.mp4"


def test_example_config_loads():
    site = load_site_config("configs/cameras/example.yaml")
    assert site.name == "demo-crossing"
    assert len(site.cameras) == 2
    assert site.cameras[0].id == "cam-north"


def test_config_rejects_duplicates(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\ncameras:\n"
        "  - {id: dup, source: '0'}\n"
        "  - {id: dup, source: '1'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate camera id"):
        load_site_config(bad)


def test_video_source_fails_cleanly_on_missing_file():
    with pytest.raises(RuntimeError, match="could not open"):
        VideoSource("no-such-file.mp4").open()


def test_overlay_keeps_frame_shape():
    vs = VideoSource(0)
    vs.fps, vs.width, vs.height = 30.0, 640, 480
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = vs.overlay(frame, 42)
    assert out.shape == frame.shape


def test_cli_parses_run():
    parser = build_parser()
    args = parser.parse_args(["run", "--source", "data/clip.mp4", "--max-frames", "5"])
    assert args.source == "data/clip.mp4"
    assert args.max_frames == 5
