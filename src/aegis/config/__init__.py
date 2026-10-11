"""Config package: YAML site/camera definitions."""

from aegis.config.schema import CameraConfig, SiteConfig, load_site_config

__all__ = ["CameraConfig", "SiteConfig", "load_site_config"]
