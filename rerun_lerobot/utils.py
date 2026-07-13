"""Utility functions for data conversion."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy.typing as npt


@contextmanager
def suppress_ffmpeg_output() -> Iterator[None]:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def unwrap_singleton(value: object) -> object:
    """Unwrap single-element lists or arrays to their scalar value."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
        return value[0]
    return value


def to_float32_vector(value: object, expected_dim: int, label: str) -> npt.NDArray[np.float32]:
    """
    Convert a value to a float32 numpy array with expected dimensions.

    Args:
        value: Input value to convert
        expected_dim: Expected dimension of the output vector
        label: Label for error messages

    Returns:
        Float32 numpy array with shape (expected_dim,)

    Raises:
        ValueError: If value is None or has incorrect dimensions

    """
    if value is None:
        raise ValueError(f"Missing {label} value.")
    value = unwrap_singleton(value)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    # Skip dimension check if expected_dim is -1 (variable length)
    if expected_dim != -1 and array.shape[0] != expected_dim:
        raise ValueError(f"{label} has dim {array.shape[0]} but expected {expected_dim}.")
    return array


def normalize_times(values: Iterable[object]) -> npt.NDArray[np.int64]:
    """
    Normalize time values to nanosecond precision int64.

    Args:
        values: Iterable of time values (datetime64, timedelta64, float, int, or
            Pandas Timestamp/Timedelta)

    Returns:
        Int64 array representing nanoseconds

    Raises:
        ValueError: If datetime values are not in nanosecond precision

    """
    times = np.asarray(list(values))

    # Handle Pandas Timestamp/Timedelta objects (both expose .value in ns)
    if times.dtype == object and len(times) > 0:
        import pandas as pd

        if isinstance(times[0], (pd.Timestamp, pd.Timedelta)):
            return np.array([t.value for t in times], dtype="int64")

    if np.issubdtype(times.dtype, np.datetime64):
        # Verify we have at least nanosecond resolution
        dt_unit = np.datetime_data(times.dtype)[0]
        if dt_unit in ("s", "ms", "us"):
            raise ValueError(
                f"Datetime values have insufficient resolution: {dt_unit}. "
                "Expected nanosecond ('ns') resolution for accurate timestamp conversion."
            )
        return times.astype("datetime64[ns]").astype("int64")
    if np.issubdtype(times.dtype, np.timedelta64):
        return times.astype("timedelta64[ns]").astype("int64")
    if np.issubdtype(times.dtype, np.floating):
        return (times * 1_000_000_000.0).astype("int64")
    return times.astype("int64")


TimeInput = np.datetime64 | np.floating[Any] | np.integer[Any] | float | int


def make_time_grid(min_value: TimeInput, max_value: TimeInput, fps: int) -> npt.NDArray[np.generic]:
    """
    Create a time grid at the specified FPS between min and max values.

    Args:
        min_value: Minimum time value
        max_value: Maximum time value
        fps: Frames per second for the grid

    Returns:
        Array of time values at regular intervals

    """
    min_array = np.asarray(min_value)
    if np.issubdtype(min_array.dtype, np.datetime64):
        step = np.timedelta64(int(1_000_000_000 / fps), "ns")
        if max_value <= min_value:
            return np.array([min_value])
        return np.arange(min_value, max_value, step)
    if max_value <= min_value:
        return np.array([min_value], dtype=np.float64)
    return np.arange(float(min_value), float(max_value), 1.0 / fps)


def get_entity_path(fully_qualified_column: str | None) -> str | None:
    """
    Extract the entity path from a fully qualified column name.

    The fully qualified column format is: "entity_path:ComponentName:field_name"
    This function extracts just the entity_path portion.

    Args:
        fully_qualified_column: Fully qualified column name (e.g., "/robot/joint_states:JointState:positions")

    Returns:
        Entity path (e.g., "/robot/joint_states"), or None if input is None

    Examples:
        >>> get_entity_path("/robot/joint_states:JointState:positions")
        "/robot/joint_states"
        >>> get_entity_path(None)
        None

    """
    if fully_qualified_column is None:
        return None
    return fully_qualified_column.split(":")[0]


def split_dataset_url(dataset_url: str) -> tuple[str, str]:
    """
    Split a Rerun dataset entry URL into a catalog server URL and an entry id.

    Args:
        dataset_url: Full Rerun dataset entry URL, e.g.
            "rerun://hostname:443/entry/18B40C6FA7631F942c0e90030ac230fa".

    Returns:
        A tuple of (catalog_url, entry_id), where catalog_url is the
        scheme+host+port and entry_id is the dataset id.

    Raises:
        ValueError: If the URL has no host or no entry id.

    Examples:
        >>> split_dataset_url("rerun://host:443/entry/abc123")
        ('rerun://host:443', 'abc123')

    """
    from urllib.parse import urlparse

    parsed = urlparse(dataset_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid dataset URL (expected e.g. 'rerun://host:port/entry/<id>'): {dataset_url}")

    parts = [part for part in parsed.path.split("/") if part]
    # Accept both `/entry/<id>` and a bare trailing `<id>`.
    if len(parts) >= 2 and parts[-2] == "entry":
        entry_id = parts[-1]
    elif len(parts) == 1:
        entry_id = parts[0]
    else:
        raise ValueError(f"Could not find an entry id in dataset URL: {dataset_url}")

    catalog_url = f"{parsed.scheme}://{parsed.netloc}"
    return catalog_url, entry_id
