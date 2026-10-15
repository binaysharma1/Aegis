"""Draw detection boxes + labels on a frame.

Pure OpenCV — no model needed, so this is cheap to unit-test.
"""

from __future__ import annotations

import cv2
import numpy as np

from aegis.perception.detector import Detection
from aegis.zones.geometry import Line

# BGR colors, one per class id (wraps around), so cars look ≠ people.
PALETTE: list[tuple[int, int, int]] = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (128, 0, 128),
]


def color_for(cls: int) -> tuple[int, int, int]:
    return PALETTE[cls % len(PALETTE)]


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of frame with one rectangle + caption per detection.

    Caption is `label conf`, or `label #id conf` when tracked. Color follows
    the track id when present (so one object keeps its color), else the class.
    """
    out = frame.copy()
    for det in detections:
        key = det.track_id if det.track_id is not None else det.cls
        color = color_for(key)
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        caption = f"{det.label} {det.confidence:.2f}"
        if det.track_id is not None:
            caption = f"{det.label} #{det.track_id} {det.confidence:.2f}"
        y = det.y1 - 8 if det.y1 - 8 > 8 else det.y1 + 18  # keep text on-screen
        cv2.putText(
            out, caption, (det.x1, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return out


def draw_line(frame: np.ndarray, line: Line) -> np.ndarray:
    """Draw the virtual line in red, with dots on its endpoints."""
    out = frame.copy()
    cv2.line(out, (line.x1, line.y1), (line.x2, line.y2), (0, 0, 255), 2)
    for pt in ((line.x1, line.y1), (line.x2, line.y2)):
        cv2.circle(out, pt, 4, (0, 0, 255), -1)
    return out


def draw_signal(frame: np.ndarray, state: str) -> np.ndarray:
    """Traffic-light widget, top-right: 3 lamps, the active one lit."""
    out = frame.copy()
    lamps = (("RED", (0, 0, 255)), ("GREEN", (0, 255, 0)), ("ORANGE", (0, 165, 255)))
    x0 = out.shape[1] - 70
    cv2.rectangle(out, (x0 - 8, 8), (x0 + 32, 108), (20, 20, 20), -1)
    for i, (name, color) in enumerate(lamps):
        cv2.circle(out, (x0 + 12, 28 + i * 30), 10, color if name == state else (60, 60, 60), -1)
    cv2.putText(
        out, state, (x0 - 8, 124),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out
