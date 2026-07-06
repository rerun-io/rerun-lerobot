"""Convert Rerun RRD recordings into LeRobot v3 datasets."""

from __future__ import annotations

from rerun_lerobot import lerobot
from rerun_lerobot.lerobot.types import LeRobotConversionConfig, VideoSpec

__version__ = "0.3.0"

__all__ = [
    "LeRobotConversionConfig",
    "VideoSpec",
    "__version__",
    "lerobot",
]
