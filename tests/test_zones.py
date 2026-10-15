"""Tests for zone geometry + line drawing (no model weights needed)."""

import numpy as np
import pytest

from aegis.cli import _parse_classes, build_parser, cmd_run
from aegis.perception.draw import draw_line
from aegis.zones import Line, LineWatcher, side


def test_side_gives_opposite_signs():
    line = Line(0, 100, 200, 100)  # horizontal line at y=100
    assert side(line, 50, 50) * side(line, 50, 150) < 0
    assert side(line, 50, 100) == 0  # exactly on the line


def test_parse_ok_and_middle():
    assert Line.parse("0,25,99,25") == Line(0, 25, 99, 25)
    assert Line.middle(800, 600) == Line(0, 300, 800, 300)
    assert Line.vertical_middle(800, 600) == Line(400, 0, 400, 600)


def test_from_spec_keywords_and_coords():
    assert Line.from_spec("middle", 800, 600) == Line(0, 300, 800, 300)
    assert Line.from_spec("vmiddle", 800, 600) == Line(400, 0, 400, 600)
    assert Line.from_spec("0,1,2,3", 800, 600) == Line(0, 1, 2, 3)
    assert Line.is_keyword("vmiddle") and not Line.is_keyword("0,1,2,3")


def test_vertical_line_crossing():
    w = LineWatcher(Line(100, 0, 100, 200))  # vertical line at x=100
    assert side(Line(100, 0, 100, 200), 50, 50) * side(Line(100, 0, 100, 200), 150, 50) < 0
    w.update(9, "person", 50, 50)
    cross = w.update(9, "person", 150, 50)
    assert cross is not None and (cross.track_id, cross.label) == (9, "person")


def test_parse_classes():
    assert _parse_classes(None) is None
    assert _parse_classes("") is None
    assert _parse_classes("person") == {"person"}
    assert _parse_classes("Person, CAR ") == {"person", "car"}


def test_parse_bad():
    with pytest.raises(ValueError, match="expected 'x1,y1,x2,y2'"):
        Line.parse("0,25,99")


def _walker():
    return LineWatcher(Line(0, 100, 200, 100))


def test_first_sighting_never_fires():
    assert _walker().update(1, "person", 50, 50) is None


def test_same_side_twice_never_fires():
    w = _walker()
    w.update(1, "person", 50, 50)
    assert w.update(1, "person", 60, 60) is None


def test_crossing_both_directions():
    w = _walker()
    w.update(1, "person", 50, 50)  # above the line: negative side
    down = w.update(1, "person", 50, 150)  # neg -> pos
    assert down is not None and down.direction == "+"
    up = w.update(1, "person", 50, 40)  # pos -> neg
    assert up is not None and up.direction == "-"
    assert (down.track_id, down.label) == (1, "person")


def test_sitting_on_line_then_leaving_is_not_a_crossing():
    w = _walker()
    w.update(1, "person", 50, 100)  # exactly on the line: stored, no event
    assert w.update(1, "person", 50, 150) is None  # prev was 0: still no event
    assert w.update(1, "person", 50, 50) is not None  # now a real flip


def test_tracks_are_independent():
    w = _walker()
    w.update(1, "person", 50, 50)
    assert w.update(2, "car", 50, 150) is None  # id 2 seen first time: no event


def test_draw_line_paints_red():
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    out = draw_line(frame, Line(0, 25, 99, 25))
    assert out.shape == frame.shape
    assert out[25, 50].tolist() == [0, 0, 255]
    assert out[0, 0].tolist() == [0, 0, 0]


def test_cli_line_flag_parses():
    args = build_parser().parse_args(["run", "--track", "--line", "middle"])
    assert args.line == "middle"
    assert build_parser().parse_args(["run"]).line is None
    person_run = build_parser().parse_args(
        ["run", "--track", "--line", "vmiddle", "--classes", "person"]
    )
    assert person_run.line == "vmiddle"
    assert person_run.classes == "person"


def test_line_without_track_fails_cleanly():
    import argparse

    args = argparse.Namespace(
        config=None, no_detect=True, line="middle", track=False,
        source="0", show=False, max_frames=0, save_frame=None,
        signal=False, cycle=5.0, model="yolov8n.pt", conf=0.4,
        imgsz=640, every=1,
    )
    assert cmd_run(args) == 1  # exits before touching any camera
