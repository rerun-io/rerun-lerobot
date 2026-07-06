"""
End-to-end tests for VideoStream-source cameras against a locally generated RRD.

Builds small recordings with an H.264 ``VideoStream`` camera (plus action/state
Scalars) and converts them, exercising the two remux code paths in
``converter.py``:

- the *fast path* (every camera is a remuxable video → no frame decoding), and
- the *mixed path* (a remuxable video alongside a re-encoded raw image).

Each recording becomes one episode, and two recordings are used so the video
files span multiple episodes — LeRobot concatenates them into a single MP4, so
this also guards against the remuxed video for a later episode clobbering an
earlier one.

Correctness is checked by decoding the written MP4s directly (offline); we avoid
loading the dataset back through ``LeRobotDataset``, which reaches out to the
Hugging Face Hub for a version check.

Runs fully offline; skips if the heavy deps are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rerun")
pytest.importorskip("lerobot")
pytest.importorskip("av")

import av  # noqa: E402
import rerun as rr  # noqa: E402

from rerun_lerobot.lerobot.export import convert_rrd_dataset_to_lerobot  # noqa: E402
from rerun_lerobot.lerobot.types import LeRobotConversionConfig, VideoSpec  # noqa: E402

NUM_FRAMES = 6
IMG_W = IMG_H = 32


def _encode_h264_samples(brightnesses: list[int]) -> list[bytes]:
    """Encode one solid-color frame per brightness into H.264 packet payloads."""
    stream = av.open("/dev/null", mode="w", format="h264").add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = IMG_W, IMG_H, "yuv420p"
    # Intra-only keeps every packet an independently decodable keyframe.
    stream.options = {"g": "1", "bf": "0"}

    samples: list[bytes] = []
    for value in brightnesses:
        arr = np.full((IMG_H, IMG_W, 3), value, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            samples.append(bytes(packet))
    for packet in stream.encode(None):  # flush
        samples.append(bytes(packet))
    return samples


def _write_recording(path: Path, recording_id: str, *, base: int, with_raw: bool) -> None:
    """Write one RRD with an H.264 VideoStream camera (+ optional raw image) and scalars."""
    rr.init("rerun_lerobot_video_fixture", recording_id=recording_id)
    rr.save(str(path))

    samples = _encode_h264_samples([(base + i * 20) % 256 for i in range(NUM_FRAMES)])
    if len(samples) != NUM_FRAMES:
        pytest.skip("encoder buffered packets; cannot align 1:1 for this fixture")

    rr.log("/cam/video", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
    for i in range(NUM_FRAMES):
        rr.set_time("frame", sequence=i)
        rr.log("/cam/video", rr.VideoStream.from_fields(sample=samples[i], is_keyframe=True))
        if with_raw:
            rr.log("/cam/raw", rr.Image(np.full((IMG_H, IMG_W, 3), (base + i * 10) % 256, dtype=np.uint8)))
        rr.log("/action", rr.Scalars([float(i), float(i + 1)]))
        rr.log("/state", rr.Scalars([float(-i)]))
    rr.disconnect()


def _config(*, with_raw: bool) -> LeRobotConversionConfig:
    videos = [VideoSpec(key="video", path="/cam/video")]
    if with_raw:
        videos.append(VideoSpec(key="raw", path="/cam/raw"))
    return LeRobotConversionConfig(
        fps=10,
        index_column="frame",
        action="/action:Scalars:scalars",
        state="/state:Scalars:scalars",
        task=None,
        videos=videos,
        output_format=None,  # keep the h264 source -> exercise remux
    )


def _decode_frame_count(mp4_path: Path) -> int:
    container = av.open(str(mp4_path))
    try:
        return sum(1 for _ in container.decode(video=0))
    finally:
        container.close()


def test_videostream_remux_fast_path(tmp_path: Path) -> None:
    """Video-only cameras take the no-decode fast path; two episodes share one MP4."""
    rrd_dir = tmp_path / "rrds"
    rrd_dir.mkdir()
    _write_recording(rrd_dir / "episode_0.rrd", "episode_0", base=0, with_raw=False)
    _write_recording(rrd_dir / "episode_1.rrd", "episode_1", base=100, with_raw=False)

    out = tmp_path / "ds"
    convert_rrd_dataset_to_lerobot(
        rrd_dir=rrd_dir, output_dir=out, dataset_name="vid", repo_id="vid", config=_config(with_raw=False)
    )

    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 2 * NUM_FRAMES
    assert info["features"]["observation.images.video"]["dtype"] == "video"

    mp4s = list(out.glob("videos/**/*.mp4"))
    assert len(mp4s) == 1, "both episodes should share a single concatenated video file"
    # All frames from both episodes must be decodable (no clobbering on episode 2).
    assert _decode_frame_count(mp4s[0]) == 2 * NUM_FRAMES


def test_videostream_mixed_remux_and_reencode(tmp_path: Path) -> None:
    """A remuxable video alongside a re-encoded raw image: both stored, both intact."""
    rrd_dir = tmp_path / "rrds"
    rrd_dir.mkdir()
    _write_recording(rrd_dir / "episode_0.rrd", "episode_0", base=0, with_raw=True)
    _write_recording(rrd_dir / "episode_1.rrd", "episode_1", base=100, with_raw=True)

    out = tmp_path / "ds"
    convert_rrd_dataset_to_lerobot(
        rrd_dir=rrd_dir, output_dir=out, dataset_name="mix", repo_id="mix", config=_config(with_raw=True)
    )

    feats = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))["features"]
    assert feats["observation.images.video"]["dtype"] == "video"
    assert feats["observation.images.raw"]["dtype"] == "video"

    for key in ("video", "raw"):
        mp4 = next(out.glob(f"videos/observation.images.{key}/**/*.mp4"))
        assert _decode_frame_count(mp4) == 2 * NUM_FRAMES, f"{key} video lost frames"
