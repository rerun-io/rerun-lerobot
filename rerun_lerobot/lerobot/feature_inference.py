"""Feature shape inference for LeRobot datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rerun_lerobot.lerobot.cameras import infer_camera_feature

if TYPE_CHECKING:
    import pyarrow as pa

    from rerun_lerobot.camera import ResolvedCamera
    from rerun_lerobot.lerobot.types import FeatureSpec, LeRobotConversionConfig


def _infer_column_dim(table: pa.Table, column: str, label: str) -> int:
    """Dimension of one column, from its first non-null value (flattened)."""
    if column not in table.column_names:
        raise ValueError(f"{label} column '{column}' not found in table. Available columns: {table.column_names}")

    values = table[column].to_pylist()
    sample = next((v for v in values if v is not None), None)
    if sample is None:
        raise ValueError(f"Could not infer {label} dimension: no non-null values found in column '{column}'")

    array = np.asarray(sample).flatten()
    # Validate that the data can be converted to float32
    if not np.can_cast(array.dtype, np.float32, "same_kind"):
        raise ValueError(f"{label} column '{column}' has dtype '{array.dtype}' which cannot be safely cast to float32")
    return len(array)


def _infer_vector_feature(
    table: pa.Table,
    columns: list[str],
    names: list[str] | None,
    label: str,
) -> FeatureSpec:
    """Infer one float32 vector feature spanning one or more columns (dims are summed)."""
    total_dim = sum(_infer_column_dim(table, column, label) for column in columns)
    if names is not None and len(names) != total_dim:
        raise ValueError(f"{label.capitalize()} names length does not match inferred {label} dimension.")
    return {"dtype": "float32", "shape": (total_dim,), "names": names}


def infer_features(
    *,
    table: pa.Table,
    config: LeRobotConversionConfig,
    cameras: list[ResolvedCamera] | None = None,
    camera_tables: dict[str, pa.Table] | None = None,
) -> dict[str, FeatureSpec]:
    """
    Infer feature specifications from a pre-queried PyArrow table.

    Args:
        table: PyArrow table containing the scalar columns (action, state, task)
        config: Conversion configuration
        cameras: Resolved cameras to infer image/video feature specs for
        camera_tables: Per-camera inference table, keyed by camera key (defaults to ``table``)

    Returns:
        Dictionary mapping feature names to their specifications

    Raises:
        ValueError: If features cannot be inferred or names don't match dimensions

    """
    features: dict[str, FeatureSpec] = {}

    features["action"] = _infer_vector_feature(table, config.action_columns, config.action_names, "action")
    features["observation.state"] = _infer_vector_feature(table, config.state_columns, config.state_names, "state")

    # Infer camera feature specs (video / encoded-image / raw-image sources).
    for camera in cameras or []:
        camera_table = camera_tables[camera.key] if camera_tables is not None else table
        try:
            features[camera.feature_key] = infer_camera_feature(camera, camera_table, camera.index_column)
        except ValueError as err:
            raise ValueError(f"Could not infer shape for camera '{camera.key}' at '{camera.path}': {err}") from err

    return features
