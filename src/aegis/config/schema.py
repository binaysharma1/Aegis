"""Typed site/camera configuration (YAML -> Pydantic)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class CameraConfig(BaseModel):
    """One physical camera at an intersection site."""

    id: str = Field(min_length=1, description="Unique camera id, e.g. 'cam-north'")
    source: str = Field(min_length=1, description="File path, RTSP URL, or index like '0'")
    fps: float = Field(default=30.0, gt=0, le=120)
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=720, gt=0)


class SiteConfig(BaseModel):
    """An intersection site: one or more cameras."""

    name: str = Field(min_length=1, description="Site name, e.g. 'kalanki-crossing'")
    cameras: list[CameraConfig] = Field(min_length=1)

    @field_validator("cameras")
    @classmethod
    def _unique_ids(cls, cameras: list[CameraConfig]) -> list[CameraConfig]:
        ids = [c.id for c in cameras]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate camera id in {ids}")
        return cameras


def load_site_config(path: str | Path) -> SiteConfig:
    """Load and validate a site YAML file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a mapping at top level, got {type(data).__name__}")
    return SiteConfig.model_validate(data)
