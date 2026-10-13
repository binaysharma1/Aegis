"""YOLO object detection (Phase 1 slice).

A *classifier* (e.g. AlexNet) answers ONE label for the whole image.
A *detector* answers MANY boxes + labels per frame — that is what we need,
so we use a COCO-pretrained YOLO nano: tiny, CPU-friendly, no training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """One object: rectangle corners in pixels, label, and certainty 0-1."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    cls: int = 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class Detector:
    """Loads YOLO weights ONCE, then answers detect(frame) per frame."""

    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.4, imgsz: int = 640) -> None:
        from ultralytics import YOLO  # lazy import: `aegis hello` stays instant

        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model)  # auto-downloads ~6MB weights on first run

    @property
    def names(self) -> dict[int, str]:
        """Class id -> label, e.g. {0: 'person', 2: 'car'} (80 COCO classes)."""
        return dict(self.model.names)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run one frame -> list of boxes (already thresholded by conf)."""
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        out: list[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cls = int(box.cls[0])
            out.append(
                Detection(x1, y1, x2, y2, self.model.names[cls], float(box.conf[0]), cls)
            )
        return out
