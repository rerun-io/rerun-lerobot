"""Feature shape inference for LeRobot datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rerun_lerobot.lerobot.cameras import infer_camera_feature

if TYPE_CHECKING:
    import pyarrow as pa

    from rerun_lerobot.camera import ResolvedCamera
    from rerun_lerobot.lerobot.types import FeatureSpec, LeRobotConversionConfig


def infer_features(
    *,
    table: pa.Table,
    config: LeRobotConversionConfig,
    cameras: list[ResolvedCamera] | None = None,
) -> dict[str, FeatureSpec]:
    """
    Infer feature specifications from a pre-queried PyArrow table.

    Args:
        table: PyArrow table containing all necessary columns (action, state, camera samples)
        config: Conversion configuration
        cameras: Resolved cameras to infer image/video feature specs for

    Returns:
        Dictionary mapping feature names to their specifications

    Raises:
        ValueError: If features cannot be inferred or names don't match dimensions

    """
    features: dict[str, FeatureSpec] = {}

    # Infer action dimension
    if config.action not in table.column_names:
        raise ValueError(f"Action column '{config.action}' not found in table. Available columns: {table.column_names}")

    action_values = table[config.action].to_pylist()
    action_sample = next((v for v in action_values if v is not None), None)
    if action_sample is None:
        raise ValueError(f"Could not infer action dimension: no non-null values found in column '{config.action}'")

    action_array = np.asarray(action_sample).flatten()
    # Validate that action data can be converted to float32
    if not np.can_cast(action_array.dtype, np.float32, "same_kind"):
        raise ValueError(f"Action data has dtype '{action_array.dtype}' which cannot be safely cast to float32")

    action_dim = len(action_array)
    if config.action_names is not None and len(config.action_names) != action_dim:
        raise ValueError("Action names length does not match inferred action dimension.")
    features["action"] = {"dtype": "float32", "shape": (action_dim,), "names": config.action_names}

    # Infer state dimension
    if config.state not in table.column_names:
        raise ValueError(f"State column '{config.state}' not found in table. Available columns: {table.column_names}")

    state_values = table[config.state].to_pylist()
    state_sample = next((v for v in state_values if v is not None), None)
    if state_sample is None:
        raise ValueError(f"Could not infer state dimension: no non-null values found in column '{config.state}'")

    state_array = np.asarray(state_sample).flatten()
    # Validate that state data can be converted to float32
    if not np.can_cast(state_array.dtype, np.float32, "same_kind"):
        raise ValueError(f"State data has dtype '{state_array.dtype}' which cannot be safely cast to float32")

    state_dim = len(state_array)
    if config.state_names is not None and len(config.state_names) != state_dim:
        raise ValueError("State names length does not match inferred state dimension.")
    features["observation.state"] = {"dtype": "float32", "shape": (state_dim,), "names": config.state_names}

    # Infer camera feature specs (video / encoded-image / raw-image sources).
    for camera in cameras or []:
        try:
            features[camera.feature_key] = infer_camera_feature(camera, table, config.index_column)
        except ValueError as err:
            raise ValueError(f"Could not infer shape for camera '{camera.key}' at '{camera.path}': {err}") from err

    return features
