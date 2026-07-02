"""Tests for video frame selection and single-pass decoding."""

from __future__ import annotations

import numpy as np
import pytest

from rerun_lerobot.lerobot.video_processing import _latest_at_indices


def test_latest_at_exact_and_between() -> None:
    frame_times = np.array([0, 100, 200, 300], dtype=np.int64)
    targets = np.array([0, 100, 150, 200, 299, 300, 999], dtype=np.int64)
    # exact hits, "between" rounds down, past-end clamps to last frame.
    assert _latest_at_indices(frame_times, targets) == [0, 1, 1, 2, 2, 3, 3]


def test_latest_at_before_first_clamps_to_zero() -> None:
    frame_times = np.array([500, 600], dtype=np.int64)
    targets = np.array([0, 100, 499], dtype=np.int64)
    assert _latest_at_indices(frame_times, targets) == [0, 0, 0]


def test_latest_at_unsorted_frame_times() -> None:
    # Frames delivered out of time order must still map correctly.
    frame_times = np.array([300, 0, 200, 100], dtype=np.int64)  # positions 0..3
    targets = np.array([0, 150, 350], dtype=np.int64)
    # t=0 -> time 0 (pos 1); t=150 -> time 100 (pos 3); t=350 -> time 300 (pos 0)
    assert _latest_at_indices(frame_times, targets) == [1, 3, 0]


def test_latest_at_empty_targets() -> None:
    assert _latest_at_indices(np.array([0, 1], dtype=np.int64), np.array([], dtype=np.int64)) == []


def _encode_h264_samples(brightnesses: list[int], width: int = 32, height: int = 32) -> list[bytes]:
    """Encode one solid-color frame per brightness into a list of H.264 packet payloads."""
    import av

    stream = av.open("/dev/null", mode="w", format="h264").add_stream("libx264", rate=30)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    # Intra-only keeps every packet independently decodable (no reordering surprises).
    stream.options = {"g": "1", "bf": "0"}

    samples: list[bytes] = []
    for value in brightnesses:
        arr = np.full((height, width, 3), value, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            samples.append(bytes(packet))
    for packet in stream.encode(None):  # flush
        samples.append(bytes(packet))
    return samples


def test_decode_video_frames_at_times_roundtrip() -> None:
    av = pytest.importorskip("av")
    assert av  # silence unused

    from rerun_lerobot.lerobot.video_processing import decode_video_frames_at_times

    brightnesses = [0, 40, 80, 120, 160]
    samples = _encode_h264_samples(brightnesses)
    if len(samples) < len(brightnesses):
        pytest.skip("encoder buffered packets; cannot align 1:1 for this assertion")

    times_ns = np.arange(len(samples), dtype=np.int64) * 1_000_000  # 1 ms apart
    # Sample before start, at each frame, and past the end.
    targets = np.array([-500, 0, 1_000_000, 2_500_000, 10_000_000], dtype=np.int64)

    frames = decode_video_frames_at_times(
        samples=samples, times_ns=times_ns, target_times_ns=targets, video_format="h264"
    )

    assert len(frames) == len(targets)
    for frame in frames:
        assert frame.ndim == 3
        assert frame.shape[2] == 3

    # Mean brightness should be non-decreasing as the target time advances (latest-at).
    means = [float(np.asarray(f).mean()) for f in frames]
    assert means == sorted(means)
    # Before-start clamps to the darkest (first) frame; past-end holds the brightest.
    assert means[0] <= means[1]
    assert means[-1] == max(means)
