"""Draw detection boxes + labels on a frame.

Pure OpenCV — no model needed, so this is cheap to unit-test.
"""

from __future__ import annotations

import cv2
import numpy as np

from aegis.perception.detector import Detection

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
    """Return a copy of frame with one rectangle + `label conf` per detection."""
    out = frame.copy()
    for det in detections:
        color = color_for(det.cls)
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        caption = f"{det.label} {det.confidence:.2f}"
        y = det.y1 - 8 if det.y1 - 8 > 8 else det.y1 + 18  # keep text on-screen
        cv2.putText(
            out, caption, (det.x1, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return out
