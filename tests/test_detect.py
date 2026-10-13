"""Tests for detection drawing + CLI flags (no model weights needed)."""

import numpy as np

from aegis.cli import build_parser
from aegis.perception import Detection, color_for, draw_detections


def _det(**kw) -> Detection:
    base = {
        "x1": 10, "y1": 10, "x2": 50, "y2": 60,
        "label": "car", "confidence": 0.9, "cls": 2,
    }
    base.update(kw)
    return Detection(**base)


def test_draw_keeps_shape_and_marks_box():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(frame, [_det()])
    assert out.shape == frame.shape
    assert out[10, 10].tolist() != [0, 0, 0]  # box corner got painted
    assert out[90, 90].tolist() == [0, 0, 0]  # far-away pixel untouched


def test_draw_empty_leaves_frame_untouched():
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    assert np.array_equal(draw_detections(frame, []), frame)


def test_color_wraps_but_stable_per_class():
    assert color_for(2) == color_for(2 + 8)  # palette wraps around
    assert isinstance(color_for(0), tuple)


def test_cli_detect_flags_defaults():
    args = build_parser().parse_args(["run"])
    assert args.model == "yolov8n.pt"
    assert args.conf == 0.4
    assert args.every == 1
    assert args.no_detect is False
    assert args.track is False


def test_cli_track_flag_parses():
    args = build_parser().parse_args(["run", "--track"])
    assert args.track is True


def test_detection_defaults_to_untracked():
    assert _det().track_id is None


def test_track_colors_follow_id_not_class():
    assert color_for(3) != color_for(4)  # distinct IDs, distinct colors
    assert color_for(3) == color_for(3)  # same ID, stable color


def test_draw_tracked_keeps_shape():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(frame, [_det(track_id=7), _det(track_id=3)])
    assert out.shape == frame.shape
    assert out[10, 10].tolist() != [0, 0, 0]
