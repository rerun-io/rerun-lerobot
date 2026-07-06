"""RRD-to-LeRobot conversion: feature inference, resampling, and dataset writing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rerun_lerobot.lerobot.types import (
    FeatureSpec,
    LeRobotConversionConfig,
    RemuxData,
    RemuxInfo,
    VideoSampleData,
    VideoSpec,
)

if TYPE_CHECKING:
    from rerun_lerobot.lerobot.converter import convert_dataframe_to_episode
    from rerun_lerobot.lerobot.export import (
        convert_catalog_dataset_to_lerobot,
        convert_dataset_to_lerobot,
        convert_dataset_url_to_lerobot,
        convert_rrd_dataset_to_lerobot,
        inspect_catalog_dataset,
        inspect_dataset,
        inspect_dataset_url,
        inspect_rrd_dataset,
    )
    from rerun_lerobot.lerobot.feature_inference import infer_features
    from rerun_lerobot.lerobot.video_processing import (
        can_remux_video,
        decode_video_frame,
        decode_video_frames_at_times,
        extract_video_samples,
        infer_video_shape,
        infer_video_shape_from_table,
        remux_video_stream,
    )

# Heavy submodules (av, datafusion, lerobot, ...) are imported lazily so that
# `import rerun_lerobot` stays cheap for callers that only need the data types.
_LAZY_ATTRS = {
    "can_remux_video": "rerun_lerobot.lerobot.video_processing",
    "convert_catalog_dataset_to_lerobot": "rerun_lerobot.lerobot.export",
    "convert_dataframe_to_episode": "rerun_lerobot.lerobot.converter",
    "convert_dataset_to_lerobot": "rerun_lerobot.lerobot.export",
    "convert_dataset_url_to_lerobot": "rerun_lerobot.lerobot.export",
    "convert_rrd_dataset_to_lerobot": "rerun_lerobot.lerobot.export",
    "decode_video_frame": "rerun_lerobot.lerobot.video_processing",
    "decode_video_frames_at_times": "rerun_lerobot.lerobot.video_processing",
    "extract_video_samples": "rerun_lerobot.lerobot.video_processing",
    "infer_features": "rerun_lerobot.lerobot.feature_inference",
    "infer_video_shape": "rerun_lerobot.lerobot.video_processing",
    "infer_video_shape_from_table": "rerun_lerobot.lerobot.video_processing",
    "inspect_catalog_dataset": "rerun_lerobot.lerobot.export",
    "inspect_dataset": "rerun_lerobot.lerobot.export",
    "inspect_dataset_url": "rerun_lerobot.lerobot.export",
    "inspect_rrd_dataset": "rerun_lerobot.lerobot.export",
    "remux_video_stream": "rerun_lerobot.lerobot.video_processing",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


__all__ = [
    "FeatureSpec",
    "LeRobotConversionConfig",
    "RemuxData",
    "RemuxInfo",
    "VideoSampleData",
    "VideoSpec",
    "can_remux_video",
    "convert_catalog_dataset_to_lerobot",
    "convert_dataframe_to_episode",
    "convert_dataset_to_lerobot",
    "convert_dataset_url_to_lerobot",
    "convert_rrd_dataset_to_lerobot",
    "decode_video_frame",
    "decode_video_frames_at_times",
    "extract_video_samples",
    "infer_features",
    "infer_video_shape",
    "infer_video_shape_from_table",
    "inspect_catalog_dataset",
    "inspect_dataset",
    "inspect_dataset_url",
    "inspect_rrd_dataset",
    "remux_video_stream",
]
