"""
End-to-end tests: convert a real Rerun recording into a LeRobot v3 dataset.

Uses the public ``animated_urdf.rrd`` sample (an animated SO-ARM100 URDF). The
recording has no scalar/video streams, so we map the ``/transforms`` archetype's
translation (3-vector) to ``action`` and its quaternion (4-vector) to
``observation.state`` — enough to exercise the full query → resample → write path.

Two sources are exercised:
- a local directory of RRD files (``convert_rrd_dataset_to_lerobot``), and
- a Rerun catalog (``convert_catalog_dataset_to_lerobot``), served here by a
  local OSS server whose URL we hand to ``CatalogClient``.

The RRD is downloaded on demand and cached under ``tests/data/`` (gitignored).
The tests skip gracefully when the heavy dependencies are missing or the sample
cannot be downloaded (e.g. offline CI).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# The conversion pulls in heavy, platform-specific deps (rerun-sdk, lerobot,
# torch, ...). Skip the whole module if any of them is unavailable.
pytest.importorskip("rerun")
pytest.importorskip("lerobot")

from rerun_lerobot.lerobot.types import LeRobotConversionConfig  # noqa: E402

RRD_URL = "https://app.rerun.io/version/0.33.0/examples/animated_urdf.rrd"
DATA_DIR = Path(__file__).parent / "data"
RRD_PATH = DATA_DIR / "animated_urdf.rrd"
DATASET_NAME = "animated_urdf"


@pytest.fixture(scope="session")
def rrd_dir() -> Path:
    """Download the sample RRD (cached) and return the directory containing it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RRD_PATH.exists():
        try:
            with urllib.request.urlopen(RRD_URL, timeout=60) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError) as err:
            pytest.skip(f"Could not download sample RRD: {err}")
        RRD_PATH.write_bytes(data)
    return DATA_DIR


def _make_config() -> LeRobotConversionConfig:
    return LeRobotConversionConfig(
        fps=10,
        index_column="log_time",
        action="/transforms:Transform3D:translation",
        state="/transforms:Transform3D:quaternion",
        task=None,
        videos=[],
    )


def _assert_valid_dataset(output_dir: Path) -> None:
    info_path = output_dir / "meta" / "info.json"
    assert info_path.is_file(), "missing meta/info.json"
    assert list(output_dir.glob("data/**/*.parquet")), "no data parquet files written"
    assert list(output_dir.glob("meta/episodes/**/*.parquet")), "no episode metadata written"

    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["fps"] == 10
    assert info["total_episodes"] == 1
    assert info["total_frames"] > 0

    features = info["features"]
    assert features["action"]["dtype"] == "float32"
    assert tuple(features["action"]["shape"]) == (3,)
    assert features["observation.state"]["dtype"] == "float32"
    assert tuple(features["observation.state"]["shape"]) == (4,)


def test_convert_rrd_dir_to_lerobot(rrd_dir: Path, tmp_path: Path) -> None:
    from rerun_lerobot.lerobot.export import convert_rrd_dataset_to_lerobot

    output_dir = tmp_path / "dataset"
    convert_rrd_dataset_to_lerobot(
        rrd_dir=rrd_dir,
        output_dir=output_dir,
        dataset_name=DATASET_NAME,
        repo_id=DATASET_NAME,
        config=_make_config(),
    )
    _assert_valid_dataset(output_dir)


def test_convert_catalog_to_lerobot(rrd_dir: Path, tmp_path: Path) -> None:
    import rerun as rr

    from rerun_lerobot.lerobot.export import convert_catalog_dataset_to_lerobot

    output_dir = tmp_path / "dataset"

    # Serve the sample with a local OSS server and treat it as a remote catalog.
    with rr.server.Server(datasets={DATASET_NAME: str(rrd_dir)}) as server:
        convert_catalog_dataset_to_lerobot(
            catalog_url=server.url(),
            dataset_name=DATASET_NAME,
            output_dir=output_dir,
            repo_id=DATASET_NAME,
            config=_make_config(),
        )
    _assert_valid_dataset(output_dir)
