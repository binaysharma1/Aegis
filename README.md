# Aegis

> **Status: 🚧 Work in progress.** Aegis is a brand-new project, scaffolded just today.
> Nothing here is functional yet — the architecture below is the plan I'm building toward.

**Aegis** is an edge-AI computer vision system for monitoring traffic intersections in
real time. The long-term goal is to detect traffic lights and rule violations, read
license plates (ANPR), and automate e-challan penalty workflows.

The project is being built in stages, starting with the perception and reasoning core
before any enforcement automation.

## Planned capabilities

- 🚦 **Traffic light detection** — read signal state (red / yellow / green) from video
- 🚗 **Vehicle & pedestrian detection** — object detection + multi-object tracking
- ⚠️ **Violation detection** — starting with red-light running, then wrong-way and others
- 🔤 **ANPR** — license plate detection + OCR *(later)*
- 📋 **E-challan workflow** — offender records and penalty automation *(much later, optional)*
- 🖥️ **Event API & dashboard** — query and review detected violations *(later)*

## Architecture

```
 VideoSource ──▶ Perception ──▶ Rules Engine ──▶ Event Store ──▶ API/Dashboard
 (file/RTSP)      detect+track     state machine     SQLite+files     (later)
                       │                │
                  Signal Timeline   Zone/ROI config
```

Design principles:

- **Signal timeline as its own artifact** — the traffic-light classifier emits a smoothed
  red/yellow/green timeline per camera; violation rules simply query it.
- **Per-camera config** — an intersection is a *site* with one or more cameras. Stop lines,
  intersection polygons, lane directions, and light boxes live in YAML, not in code.
- **Video is just a timestamped stream** — a file and an RTSP feed are interchangeable, so
  development happens offline against recorded clips.
- **Models are swappable** — start with PyTorch/Ultralytics, keep an ONNX export path for
  edge deployment later.

## Project structure

```
src/aegis/
  cli.py                 # command-line entry point
  config/                # typed camera/rule configuration
  video/source.py        # VideoSource: file + RTSP
  perception/            # detector, traffic-light state, tracker
  zones/geometry.py      # polygons, line-crossing, direction
  rules/                 # violation rules + engine
  events/                # violation model, store, evidence
  pipeline.py            # wires the stages together
configs/cameras/*.yaml
tests/
```

## Tech stack (planned)

| Concern            | Choice                                  |
| ------------------ | --------------------------------------- |
| Language           | Python 3.12                             |
| Package manager    | [uv](https://github.com/astral-sh/uv)   |
| Video / CV         | OpenCV, NumPy                           |
| Detection          | Ultralytics YOLO                        |
| Tracking           | ByteTrack / BoT-SORT via `supervision`  |
| Config             | Pydantic + YAML                         |
| Storage            | SQLite + evidence files                 |
| API (later)        | FastAPI                                 |
| OCR / ANPR (later) | PaddleOCR + plate detector              |

> ⚠️ Ultralytics YOLO is AGPL-3.0 licensed. Fine for this learning/portfolio project, but
> would need swapping for a permissive or commercial detector if Aegis ever ships.

## Roadmap

- [x] **Phase 0 — Skeleton**: config loader, `VideoSource`, CLI, debug overlay
- [ ] **Phase 1 — Perception core**: detector, traffic-light classifier, tracker, zones
- [ ] **Phase 2 — Rule engine (v0)**: signal timeline + red-light-running + evidence
- [ ] **Phase 3 — Serving (v1)**: event store, FastAPI, minimal dashboard
- [ ] **Phase 4 — ANPR (v2)**: plate detection + OCR + review queue
- [ ] **Phase 5 — Edge deployment**: ONNX/TensorRT export, benchmark, hardware
- [ ] **Phase 6 — Challan workflow**: offender records, penalty automation *(optional)*

## Getting started

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

```bash
# clone
git clone <your-repo-url> aegis
cd aegis

# sync dependencies
uv sync

# sanity check
uv run aegis hello

# validate the example site config
uv run aegis config --path configs/cameras/example.yaml

# open a video source with a debug overlay (file / webcam index / RTSP)
uv run aegis run --source 0 --show
uv run aegis run --source data/clip.mp4 --show

# run the tiny events API (in-memory, Phase 3 preview)
uv run aegis serve
# -> http://127.0.0.1:8000/docs
```

## Privacy & legal note

License plates are personally identifiable information. Even during development, evidence
should be retained under a clear retention policy, encrypted at rest, and blurred in any
public demo. Automated enforcement (e-challan) carries real legal requirements and is out
of scope for now.

## License

TBD.
