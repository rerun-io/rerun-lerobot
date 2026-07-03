"""
End-to-end tests for image-source cameras against a locally generated RRD.

Builds a small recording with three camera archetypes — EncodedImage (JPEG),
EncodedImage (PNG), and raw Image — plus action/state Scalars, then converts it
with different ``--output-format`` choices and checks the resulting LeRobot dataset.

Runs fully offline; skips if the heavy deps are missing.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rerun")
pytest.importorskip("lerobot")

import rerun as rr  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from rerun_lerobot.lerobot.export import convert_rrd_dataset_to_lerobot  # noqa: E402
from rerun_lerobot.lerobot.types import LeRobotConversionConfig, VideoSpec  # noqa: E402

NUM_FRAMES = 6
IMG_H, IMG_W = 16, 24


@pytest.fixture(scope="session")
def image_rrd_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate an RRD with jpeg/png/raw cameras + action/state scalars."""
    out_dir = tmp_path_factory.mktemp("image_rrd")
    rr.init("rerun_lerobot_fixture", recording_id="episode_0")
    rr.save(str(out_dir / "episode_0.rrd"))
    for i in range(NUM_FRAMES):
        rr.set_time("frame", sequence=i)
        arr = np.full((IMG_H, IMG_W, 3), (i * 30) % 256, dtype=np.uint8)

        jpeg = io.BytesIO()
        PILImage.fromarray(arr).save(jpeg, format="JPEG")
        rr.log("/cam/jpeg", rr.EncodedImage(contents=jpeg.getvalue(), media_type="image/jpeg"))

        png = io.BytesIO()
        PILImage.fromarray(arr).save(png, format="PNG")
        rr.log("/cam/png", rr.EncodedImage(contents=png.getvalue(), media_type="image/png"))

        rr.log("/cam/raw", rr.Image(arr))

        rr.log("/action", rr.Scalars([float(i), float(i + 1)]))
        rr.log("/state", rr.Scalars([float(-i)]))
    rr.disconnect()
    return out_dir


def _config(output_format: str | None) -> LeRobotConversionConfig:
    return LeRobotConversionConfig(
        fps=10,
        index_column="frame",
        action="/action:Scalars:scalars",
        state="/state:Scalars:scalars",
        task=None,
        videos=[
            VideoSpec(key="jpeg", path="/cam/jpeg"),
            VideoSpec(key="png", path="/cam/png"),
            VideoSpec(key="raw", path="/cam/raw"),
        ],
        output_format=output_format,
    )


def test_output_format_png_stores_all_as_images(image_rrd_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "ds_png"
    convert_rrd_dataset_to_lerobot(
        rrd_dir=image_rrd_dir, output_dir=out, dataset_name="fix", repo_id="fix", config=_config("png")
    )
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    feats = info["features"]
    for key in ("observation.images.jpeg", "observation.images.png", "observation.images.raw"):
        assert feats[key]["dtype"] == "image"
        assert tuple(feats[key]["shape"]) == (IMG_H, IMG_W, 3)
    assert info["total_frames"] == NUM_FRAMES
    # Image-dtype frames are embedded inline in the data parquet (no video files).
    assert list(out.glob("data/**/*.parquet")), "expected data parquet"
    assert not list(out.glob("videos/**/*.mp4")), "should not have written video"


def test_keep_original_mixes_image_and_video(image_rrd_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "ds_keep"
    convert_rrd_dataset_to_lerobot(
        rrd_dir=image_rrd_dir, output_dir=out, dataset_name="fix", repo_id="fix", config=_config(None)
    )
    feats = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))["features"]
    # PNG source is kept as an image; JPEG and raw fall back to h264 video.
    assert feats["observation.images.png"]["dtype"] == "image"
    assert feats["observation.images.jpeg"]["dtype"] == "video"
    assert feats["observation.images.raw"]["dtype"] == "video"
    assert list(out.glob("videos/**/*.mp4")), "expected encoded video for jpeg/raw"
