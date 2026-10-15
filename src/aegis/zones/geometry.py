"""Zone geometry: virtual lines in pixel space (software sensors).

A physical road sensor (inductive loop, IR beam) answers "something passed
this spot". A virtual line is the camera-side equivalent: "a track's center
moved from one side of this line to the other". Same question, no hardware.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Line:
    """A straight segment in pixels: (x1, y1) -> (x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def parse(cls, raw: str) -> Line:
        """Parse 'x1,y1,x2,y2' (the --line flag format)."""
        try:
            x1, y1, x2, y2 = (int(v) for v in raw.split(","))
        except ValueError:
            raise ValueError(f"bad --line {raw!r}: expected 'x1,y1,x2,y2'") from None
        return cls(x1, y1, x2, y2)

    @classmethod
    def middle(cls, width: int, height: int) -> Line:
        """Horizontal line across the middle of a width x height frame."""
        return cls(0, height // 2, width, height // 2)

    @classmethod
    def vertical_middle(cls, width: int, height: int) -> Line:
        """Vertical line down the middle — for side-to-side walkers."""
        return cls(width // 2, 0, width // 2, height)

    @classmethod
    def is_keyword(cls, raw: str) -> bool:
        """True for 'middle' (horizontal) and 'vmiddle' (vertical) shortcuts."""
        return raw.strip().lower() in ("middle", "vmiddle", "vertical-middle")

    @classmethod
    def from_spec(cls, raw: str, width: int, height: int) -> Line:
        """Resolve a --line spec: keyword or 'x1,y1,x2,y2', against frame size."""
        key = raw.strip().lower()
        if key == "middle":
            return cls.middle(width, height)
        if key in ("vmiddle", "vertical-middle"):
            return cls.vertical_middle(width, height)
        return cls.parse(raw)


def side(line: Line, px: float, py: float) -> float:
    """Which side of the line is point (px, py)? Only the sign matters.

    This is the z-component of (line direction) x (line start -> point):
    positive = left of the line's direction, negative = right, 0 = on it.
    """
    return (line.x2 - line.x1) * (py - line.y1) - (line.y2 - line.y1) * (px - line.x1)


@dataclass
class Crossing:
    track_id: int
    label: str
    direction: str  # "+" (neg side -> pos side) or "-" (pos side -> neg side)


class LineWatcher:
    """Watches one line; emits a Crossing when a track's center changes sides.

    Remembers each track's last side, so a first sighting, lingering ON the
    line, or repeated sightings on one side never raise false events.
    Needs stable track ids — that is why --line requires --track.
    """

    def __init__(self, line: Line) -> None:
        self.line = line
        self._last: dict[int, float] = {}

    def update(self, track_id: int, label: str, cx: float, cy: float) -> Crossing | None:
        now = side(self.line, cx, cy)
        prev = self._last.get(track_id)
        self._last[track_id] = now
        if prev is None or prev == 0 or now == 0:
            return None  # first sighting, or sitting exactly on the line
        if (prev < 0) == (now < 0):
            return None  # same side: no crossing
        return Crossing(track_id, label, "+" if now > 0 else "-")
