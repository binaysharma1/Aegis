"""Perception package: detector + box drawing (tracking comes later)."""

from aegis.perception.detector import Detection, Detector
from aegis.perception.draw import color_for, draw_detections

__all__ = ["Detection", "Detector", "color_for", "draw_detections"]
