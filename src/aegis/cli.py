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

import cv2

from aegis.config import load_site_config
from aegis.video import VideoSource


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
    run.set_defaults(func=cmd_run)

    cfg = sub.add_parser("config", help="Validate a site/camera YAML config.")
    cfg.add_argument("--path", required=True, help="Path to site YAML file.")
    cfg.set_defaults(func=cmd_config)

    return parser


def cmd_hello(_args: argparse.Namespace) -> int:
    from aegis import __version__

    print(f"Aegis v{__version__} — skeleton OK. Try `aegis run --help`.")
    return 0


def _coerce_source(raw: str) -> int | str:
    """'0' -> 0 (webcam), anything else stays a path/URL string."""
    raw = raw.strip()
    return int(raw) if raw.isdigit() else raw


def cmd_run(args: argparse.Namespace) -> int:
    if args.config:
        site = load_site_config(args.config)
        print(f"Loaded site '{site.name}' with {len(site.cameras)} camera(s).")

    source = _coerce_source(args.source)
    frames = 0
    try:
        with VideoSource(source) as vs:
            print(f"Opened {vs.describe()} — press 'q' to quit.")
            for frame_no, _t, frame in vs.frames():
                frames += 1
                overlay = vs.overlay(frame, frame_no)
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
