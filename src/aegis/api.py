"""Tiny FastAPI backend for Aegis (day-2 slice).

In-memory event log only — SQLite comes in Phase 3 proper.
Run:  uv run aegis serve
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from datetime import datetime
from itertools import count
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Aegis API", version="0.1.0")


class EventIn(BaseModel):
    """What the pipeline (later) posts when a rule fires."""

    camera_id: str = Field(min_length=1)
    kind: str = Field(min_length=1, examples=["red-light-run"])
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Event(EventIn):
    id: int
    created_at: datetime


_store: list[Event] = []
_ids = count(1)
_lock = Lock()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "events": len(_store)}


@app.get("/events", response_model=list[Event])
def list_events() -> list[Event]:
    with _lock:
        return list(_store)


@app.post("/events", response_model=Event, status_code=201)
def create_event(body: EventIn) -> Event:
    with _lock:
        event = Event(
            id=next(_ids),
            created_at=datetime.now(),
            **body.model_dump(),
        )
        _store.append(event)
        return event


@app.get("/events/{event_id}", response_model=Event)
def get_event(event_id: int) -> Event:
    with _lock:
        for event in _store:
            if event.id == event_id:
                return event
    raise HTTPException(status_code=404, detail="event not found")


def reset_store() -> None:
    """Test helper: clear the in-memory log and restart ids."""
    global _ids
    with _lock:
        _store.clear()
        _ids = count(1)
