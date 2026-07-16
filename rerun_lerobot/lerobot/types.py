"""Data types for RRD to LeRobot conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

import numpy as np
import numpy.typing as npt

from rerun_lerobot.utils import get_entity_path

VideoSampleData = tuple[list[bytes], npt.NDArray[np.int64]]


class FeatureSpec(TypedDict):
    """Typed feature specification for LeRobot datasets."""

    dtype: str
    shape: tuple[int, ...]
    names: list[str] | None


class RemuxInfo(TypedDict):
    """Typed remuxing details for a single video stream."""

    samples: list[bytes]
    times_ns: npt.NDArray[np.int64]
    source_fps: float


class RemuxData(TypedDict):
    """Typed remuxing payload passed between conversion steps."""

    specs: list[VideoSpec]
    remux_info: dict[str, RemuxInfo]
    fps: int


class VideoSpec(TypedDict):
    """Specification for a video stream in the dataset."""

    key: str
    path: str
    video_format: NotRequired[str]
    index: NotRequired[str]


@dataclass(frozen=True)
class LeRobotConversionConfig:
    """Configuration for converting RRD data to LeRobot format."""

    # Output configuration
    fps: int
    index_column: str

    # Column specifications; a list of columns is concatenated per row, in order.
    action: str | list[str]
    state: str | list[str]
    task: str | None

    # Camera specifications (video streams and/or image streams).
    videos: list[VideoSpec]

    # Requested output format for all cameras: one of "png", "h264", "hevc", "av1".
    # None means "keep each camera's source format if LeRobot can store it, else h264".
    output_format: str | None = None

    # Feature names
    action_names: list[str] | None = None
    state_names: list[str] | None = None

    # Task configuration
    task_default: str = "task"

    @property
    def action_columns(self) -> list[str]:
        """The action columns as a list."""
        return [self.action] if isinstance(self.action, str) else list(self.action)

    @property
    def state_columns(self) -> list[str]:
        """The state columns as a list."""
        return [self.state] if isinstance(self.state, str) else list(self.state)

    @property
    def reference_column(self) -> str:
        """The column whose non-null rows define the output rows."""
        return self.action_columns[0]

    def camera_index_column(self, spec: VideoSpec) -> str:
        """The timeline a camera's frames are queried on (its own index, or the config's)."""
        return spec.get("index", self.index_column)

    def get_filter_list(self) -> tuple[list[str], str | None]:
        """
        Get the list of entity paths to filter and the reference path for time alignment.

        Cameras are not included; each gets its own query on its own index timeline.

        Returns:
            A tuple of (contents, reference_path) where:
            - contents: List of unique entity paths to include in the scalar query
            - reference_path: The entity path to use as reference for time alignment (action or state)

        """
        contents: list[str] = []
        reference_path: str | None = None

        for column in [*self.action_columns, *self.state_columns]:
            entity_path = get_entity_path(column)
            if entity_path is None:
                continue
            if entity_path not in contents:
                contents.append(entity_path)
            if reference_path is None:
                reference_path = entity_path

        if self.task:
            entity_path = get_entity_path(self.task)
            if entity_path is not None and entity_path not in contents:
                contents.append(entity_path)

        return contents, reference_path
