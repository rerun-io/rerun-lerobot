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
    from rerun_lerobot.lerobot.converter import apply_remuxed_videos, convert_dataframe_to_episode
    from rerun_lerobot.lerobot.feature_inference import infer_features
    from rerun_lerobot.lerobot.video_processing import (
        can_remux_video,
        decode_video_frame,
        extract_video_samples,
        infer_video_shape,
        infer_video_shape_from_table,
        remux_video_stream,
    )

# Heavy submodules (av, datafusion, lerobot, ...) are imported lazily so that
# `import rerun_lerobot` stays cheap for callers that only need the data types.
_LAZY_ATTRS = {
    "apply_remuxed_videos": "rerun_lerobot.lerobot.converter",
    "can_remux_video": "rerun_lerobot.lerobot.video_processing",
    "convert_dataframe_to_episode": "rerun_lerobot.lerobot.converter",
    "decode_video_frame": "rerun_lerobot.lerobot.video_processing",
    "extract_video_samples": "rerun_lerobot.lerobot.video_processing",
    "infer_features": "rerun_lerobot.lerobot.feature_inference",
    "infer_video_shape": "rerun_lerobot.lerobot.video_processing",
    "infer_video_shape_from_table": "rerun_lerobot.lerobot.video_processing",
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
    "apply_remuxed_videos",
    "can_remux_video",
    "convert_dataframe_to_episode",
    "decode_video_frame",
    "extract_video_samples",
    "infer_features",
    "infer_video_shape",
    "infer_video_shape_from_table",
    "remux_video_stream",
]
