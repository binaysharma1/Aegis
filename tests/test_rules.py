"""Tests for the simulated traffic light (no model weights needed)."""

import numpy as np
import pytest

from aegis.cli import build_parser
from aegis.perception.draw import draw_signal
from aegis.rules import ORDER, SimSignal


def test_cycle_order_and_period():
    sig = SimSignal(cycle=5.0)
    assert ORDER == ("RED", "GREEN", "ORANGE")
    assert sig.state_at(0) == "RED"
    assert sig.state_at(4.9) == "RED"
    assert sig.state_at(5) == "GREEN"
    assert sig.state_at(10) == "ORANGE"
    assert sig.state_at(15) == "RED"  # wraps around


def test_custom_cycle():
    sig = SimSignal(cycle=2.0)
    assert sig.state_at(2) == "GREEN"
    assert sig.state_at(6) == "RED"


def test_bad_cycle_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        SimSignal(cycle=0)


def test_state_now_starts_red():
    assert SimSignal().state_now() == "RED"  # clock just started


def test_draw_signal_keeps_shape_and_paints():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    out = draw_signal(frame, "RED")
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)  # widget painted something


def test_cli_signal_flags():
    args = build_parser().parse_args(["run", "--signal", "--cycle", "3"])
    assert args.signal is True
    assert args.cycle == 3
    plain = build_parser().parse_args(["run"])
    assert plain.signal is False and plain.cycle == 5.0
