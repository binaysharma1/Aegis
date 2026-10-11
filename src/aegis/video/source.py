"""VideoSource: file / webcam / RTSP as a timestamped frame stream."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


class VideoSource:
    """Thin wrapper over cv2.VideoCapture with timestamps + overlay.

    A file and an RTSP feed are interchangeable here on purpose:
    develop offline against recorded clips, swap in RTSP later.
    """

    def __init__(self, source: int | str) -> None:
        self.source = source
        self._cap: cv2.VideoCapture | None = None
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0

    def describe(self) -> str:
        return f"camera {self.source}" if isinstance(self.source, int) else f"'{self.source}'"

    def open(self) -> VideoSource:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open video source {self.describe()} "
                "(missing file? webcam busy? bad RTSP URL?)"
            )
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoSource:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (frame_no, timestamp_s, frame_bgr) until the stream ends."""
        if self._cap is None:
            raise RuntimeError("VideoSource.frames() called before open()")
        frame_no = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame_no, frame_no / self.fps, frame
            frame_no += 1

    def overlay(self, frame: np.ndarray, frame_no: int) -> np.ndarray:
        """Draw the Phase-0 debug overlay: frame counter, timestamp, resolution."""
        out = frame.copy()
        text = f"aegis | frame {frame_no} | t={frame_no / self.fps:.2f}s | {self.width}x{self.height}"
        cv2.rectangle(out, (0, 0), (len(text) * 9 + 16, 30), (0, 0, 0), -1)
        cv2.putText(
            out, text, (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )
        return out
