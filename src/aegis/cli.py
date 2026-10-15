"""Command-line entry point for Aegis.

Today this is Phase 0: open a video source (file / webcam / RTSP),
draw a debug overlay, and optionally validate a site config.

Examples:
    uv run aegis hello
    uv run aegis run --source 0 --show
    uv run aegis run --source data/clip.mp4 --show
    uv run aegis config --path configs/cameras/example.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import cv2

from aegis.config import load_site_config
from aegis.perception import Detector, draw_detections
from aegis.perception.draw import draw_line, draw_signal
from aegis.rules import SimSignal
from aegis.video import VideoSource
from aegis.zones import Line, LineWatcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Aegis — traffic intersection vision (Phase 0 skeleton).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    hello = sub.add_parser("hello", help="Sanity check: print version and exit.")
    hello.set_defaults(func=cmd_hello)

    run = sub.add_parser("run", help="Open a video source with a debug overlay.")
    run.add_argument(
        "--source",
        default="0",
        help="Video source: file path, RTSP URL, or camera index (default: 0).",
    )
    run.add_argument(
        "--show",
        action="store_true",
        help="Show a live window with frame overlay (needs a display).",
    )
    run.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run until end / Ctrl-C).",
    )
    run.add_argument(
        "--config",
        default=None,
        help="Optional site YAML to validate before running.",
    )
    run.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights to detect with (COCO-pretrained, default: yolov8n.pt).",
    )
    run.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Keep detections at/above this confidence 0-1 (default: 0.4).",
    )
    run.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Model input size in px; smaller = faster (default: 640).",
    )
    run.add_argument(
        "--every",
        type=int,
        default=1,
        help="Run detection every Nth frame; others reuse last boxes (default: 1).",
    )
    run.add_argument(
        "--no-detect",
        action="store_true",
        help="Skip detection entirely (video overlay only).",
    )
    run.add_argument(
        "--track",
        action="store_true",
        help="Track objects with stable IDs across frames (ByteTrack).",
    )
    run.add_argument(
        "--line",
        default=None,
        help="Virtual line: 'x1,y1,x2,y2' in pixels, 'middle' for the frame's "
        "horizontal mid-line, or 'vmiddle' for the vertical mid-line. "
        "Needs --track. Crossings are reported.",
    )
    run.add_argument(
        "--signal",
        action="store_true",
        help="Show a simulated traffic light (RED->GREEN->ORANGE). "
        "Crossing on RED counts as a violation.",
    )
    run.add_argument(
        "--classes",
        default=None,
        help="Only these labels count, e.g. 'person' or 'person,car'. "
        "Default: every class counts.",
    )
    run.add_argument(
        "--cycle",
        type=float,
        default=5.0,
        help="Seconds per light state for --signal (default: 5).",
    )
    run.add_argument(
        "--save-frame",
        default=None,
        help="Save the last annotated frame to this image path.",
    )
    run.set_defaults(func=cmd_run)

    cfg = sub.add_parser("config", help="Validate a site/camera YAML config.")
    cfg.add_argument("--path", required=True, help="Path to site YAML file.")
    cfg.set_defaults(func=cmd_config)

    serve = sub.add_parser("serve", help="Run the tiny FastAPI backend.")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    serve.set_defaults(func=cmd_serve)

    return parser


def cmd_hello(_args: argparse.Namespace) -> int:
    from aegis import __version__

    print(f"Aegis v{__version__} — skeleton OK. Try `aegis run --help`.")
    return 0


def _coerce_source(raw: str) -> int | str:
    """'0' -> 0 (webcam), anything else stays a path/URL string."""
    raw = raw.strip()
    return int(raw) if raw.isdigit() else raw


def _parse_classes(raw: str | None) -> set[str] | None:
    """'person' -> {'person'}; None/'' -> None (every class counts)."""
    if raw is None:
        return None
    wants = {c.strip().lower() for c in raw.split(",") if c.strip()}
    return wants or None


def _describe(detections: list, tracked: bool) -> str:
    """One-line summary for the periodic frame log."""
    if not detections:
        return "nothing"
    if tracked:
        ids = sorted({(d.label, d.track_id) for d in detections if d.track_id is not None})
        return ", ".join(f"{label}#{tid}" for label, tid in ids) or "untracked"
    return str(dict(Counter(d.label for d in detections)))


def cmd_run(args: argparse.Namespace) -> int:
    if args.config:
        site = load_site_config(args.config)
        print(f"Loaded site '{site.name}' with {len(site.cameras)} camera(s).")

    detector: Detector | None = None
    if not args.no_detect:
        print(f"Loading detection model '{args.model}' (first run downloads ~6MB)...")
        detector = Detector(model=args.model, conf=args.conf, imgsz=args.imgsz)
        print(f"Model ready — {len(detector.names)} COCO classes.")

    pending_line = args.line
    if pending_line is not None:
        if not args.track:
            print("error: --line needs --track (crossings need stable IDs)", file=sys.stderr)
            return 1
        if not Line.is_keyword(pending_line):
            try:
                Line.parse(pending_line)  # fail fast on bad format, before opening video
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1

    wanted = _parse_classes(args.classes)

    try:
        signal = SimSignal(cycle=args.cycle) if args.signal else None
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    source = _coerce_source(args.source)
    frames = 0
    detections: list = []
    crossings: list[dict] = []  # every crossing, as plain dicts
    watcher: LineWatcher | None = None
    last = None  # last annotated frame, for --save-frame
    try:
        with VideoSource(source) as vs:
            print(f"Opened {vs.describe()} — press 'q' to quit.")
            for frame_no, _t, frame in vs.frames():
                frames += 1
                if detector is not None and frame_no % args.every == 0:
                    if args.track:
                        detections = detector.track(frame)
                    else:
                        detections = detector.detect(frame)
                if pending_line is not None and watcher is None:
                    # resolved on the first frame: keywords need the real size
                    h, w = frame.shape[:2]
                    try:
                        line = Line.from_spec(pending_line, w, h)
                    except ValueError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                        return 1
                    watcher = LineWatcher(line)
                    print(f"Watching line ({line.x1},{line.y1})-({line.x2},{line.y2})")
                light = signal.state_now() if signal is not None else "OFF"
                if watcher is not None:
                    for d in detections:
                        if d.track_id is None:
                            continue
                        if wanted is not None and d.label.lower() not in wanted:
                            continue  # e.g. bus/car boxes ignored with --classes person
                        cross = watcher.update(
                            d.track_id, d.label,
                            (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2,
                        )
                        if cross is not None:
                            rec = {
                                "id": cross.track_id,
                                "class": cross.label,
                                "direction": cross.direction,
                                "frame": frame_no,
                                "signal": light,
                                "violation": light == "RED",
                            }
                            crossings.append(rec)
                            tag = "VIOLATION" if rec["violation"] else "CROSSED"
                            print(f"{tag} {rec}")
                annotated = draw_detections(frame, detections) if detector is not None else frame
                if watcher is not None:
                    annotated = draw_line(annotated, watcher.line)
                overlay = vs.overlay(annotated, frame_no)
                if signal is not None:
                    overlay = draw_signal(overlay, light)
                last = overlay
                if detector is not None and frames % 30 == 0:
                    print(f"frame {frame_no}: {_describe(detections, args.track)}")
                if args.show:
                    cv2.imshow("aegis — debug (q to quit)", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if args.max_frames and frames >= args.max_frames:
                    break
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.show:
            cv2.destroyAllWindows()

    print(f"Done — processed {frames} frame(s).")
    print(f"Crossings: {len(crossings)}")
    for rec in crossings:
        print(f"  {rec}")
    violations = [rec for rec in crossings if rec["violation"]]
    print(f"Violations (crossed on RED): {len(violations)}")
    for rec in violations:
        print(f"  {rec}")
    if args.save_frame and last is not None:
        cv2.imwrite(args.save_frame, last)
        print(f"Saved annotated frame -> {args.save_frame}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    try:
        site = load_site_config(args.path)
    except Exception as exc:  # noqa: BLE001 — show a clean message, not a traceback
        print(f"invalid config: {exc}", file=sys.stderr)
        return 1
    print(f"OK: site '{site.name}' with {len(site.cameras)} camera(s):")
    for cam in site.cameras:
        print(f"  - {cam.id}: source={cam.source} fps={cam.fps} size={cam.width}x{cam.height}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Serving Aegis API on http://{args.host}:{args.port} (Ctrl-C to stop)")
    uvicorn.run("aegis.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
