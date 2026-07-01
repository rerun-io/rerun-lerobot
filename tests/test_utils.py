"""Tests for the dependency-light helpers (no rerun-sdk / lerobot required)."""

from __future__ import annotations

import numpy as np
import pytest

from rerun_lerobot.lerobot.types import LeRobotConversionConfig, VideoSpec
from rerun_lerobot.utils import (
    get_entity_path,
    make_time_grid,
    normalize_times,
    to_float32_vector,
    unwrap_singleton,
)


def test_unwrap_singleton() -> None:
    assert unwrap_singleton([42]) == 42
    assert unwrap_singleton([1, 2]) == [1, 2]
    assert unwrap_singleton([]) == []
    assert unwrap_singleton(np.array([7])) == 7
    np.testing.assert_array_equal(unwrap_singleton(np.array([1, 2])), np.array([1, 2]))
    assert unwrap_singleton(3.14) == 3.14


def test_to_float32_vector_scalar_and_dims() -> None:
    out = to_float32_vector(5, expected_dim=1, label="x")
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([5.0], dtype=np.float32))

    out = to_float32_vector([[1, 2, 3]], expected_dim=3, label="x")
    np.testing.assert_array_equal(out, np.array([1, 2, 3], dtype=np.float32))

    # -1 disables the dimension check
    out = to_float32_vector([1, 2, 3, 4], expected_dim=-1, label="x")
    assert out.shape == (4,)


def test_to_float32_vector_errors() -> None:
    with pytest.raises(ValueError, match="Missing x value"):
        to_float32_vector(None, expected_dim=1, label="x")
    with pytest.raises(ValueError, match="expected 2"):
        to_float32_vector([1, 2, 3], expected_dim=2, label="x")


def test_normalize_times_floats_to_nanoseconds() -> None:
    out = normalize_times([0.0, 1.0, 2.5])
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, np.array([0, 1_000_000_000, 2_500_000_000]))


def test_normalize_times_datetime_requires_ns() -> None:
    coarse = np.array(["2020-01-01"], dtype="datetime64[s]")
    with pytest.raises(ValueError, match="insufficient resolution"):
        normalize_times(coarse)

    fine = np.array([0, 1], dtype="datetime64[ns]")
    np.testing.assert_array_equal(normalize_times(fine), np.array([0, 1]))


def test_make_time_grid_float() -> None:
    grid = make_time_grid(0.0, 1.0, fps=4)
    np.testing.assert_allclose(grid, [0.0, 0.25, 0.5, 0.75])

    # Degenerate range collapses to a single point.
    np.testing.assert_array_equal(make_time_grid(1.0, 1.0, fps=4), np.array([1.0]))


def test_make_time_grid_datetime() -> None:
    lo = np.datetime64(0, "ns")
    hi = np.datetime64(1_000_000_000, "ns")  # 1 second
    grid = make_time_grid(lo, hi, fps=2)
    assert grid.shape == (2,)


def test_get_entity_path() -> None:
    assert get_entity_path("/robot/action:Scalars:scalars") == "/robot/action"
    assert get_entity_path(None) is None


def test_config_get_filter_list() -> None:
    config = LeRobotConversionConfig(
        fps=10,
        index_column="real_time",
        action="/robot/action:Scalars:scalars",
        state="/robot/state:Scalars:scalars",
        task="/task:TextDocument:text",
        videos=[VideoSpec(key="front", path="/camera/front")],
    )
    contents, reference_path = config.get_filter_list()
    assert reference_path == "/robot/action"
    assert contents == [
        "/robot/action",
        "/robot/state",
        "/task",
        "/camera/front",
    ]
