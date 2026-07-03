"""
Resolve camera sources and extract frames across all supported archetypes.

Bridges the pure detection/decoding helpers in :mod:`rerun_lerobot.camera` with the
Arrow/AV data plumbing: it probes each camera's source codec, resolves the output
format, infers the LeRobot feature spec, and extracts per-row frames (decoding
EncodedImage/Image, or decoding VideoStream via the single-pass video decoder).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from rerun_lerobot import camera as cam
from rerun_lerobot.camera import ResolvedCamera
from rerun_lerobot.lerobot.video_processing import (
    _latest_at_indices,
    decode_video_frames_at_times,
    extract_first_video_sample,
)
from rerun_lerobot.utils import normalize_times, unwrap_singleton

if TYPE_CHECKING:
    from rerun_lerobot.lerobot.types import FeatureSpec, VideoSpec


def _first_non_null(column: pa.ChunkedArray) -> object:
    for value in column.to_pylist():
        if value is not None:
            return value
    return None


def _blob_to_bytes(blob: object) -> bytes:
    """A Rerun blob column yields a list-of-lists of uint8; take the first instance."""
    value = blob
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return bytes(bytearray(cast("list[int]", value)))


def probe_video_codec(sample_bytes: bytes) -> str | None:
    """Read the codec name (h264/hevc/av1/...) from a raw VideoStream packet."""
    import av

    for fmt in ("h264", "hevc", "av1"):
        try:
            container = av.open(io.BytesIO(sample_bytes), format=fmt, mode="r")
            name = str(container.streams.video[0].codec_context.name)
            container.close()
            return {"h265": "hevc"}.get(name, name)
        except Exception:
            continue
    return None


def resolve_cameras(
    *,
    videos: list[VideoSpec],
    schema_names: list[str],
    inference_table: pa.Table,
    requested_format: str | None,
) -> list[ResolvedCamera]:
    """
    Resolve every camera spec to its source kind, source codec, and output format.

    Raises:
        ValueError: If a camera has no supported archetype or an unusable format.

    """
    requested = cam.validate_output_format(requested_format)
    resolved: list[ResolvedCamera] = []
    for spec in videos:
        path = spec["path"]
        kind = cam.detect_camera_kind(schema_names, path)

        source_codec: str | None = None
        if kind == "video":
            sample_col = f"{path}:VideoStream:sample"
            sample = (
                _first_non_null(inference_table[sample_col]) if sample_col in inference_table.column_names else None
            )
            if sample is not None:
                source_codec = probe_video_codec(_blob_to_bytes(sample))
            if source_codec is None:
                source_codec = spec.get("video_format", "h264")
        elif kind == "encoded_image":
            blob_col = f"{path}:EncodedImage:blob"
            sample = _first_non_null(inference_table[blob_col]) if blob_col in inference_table.column_names else None
            if sample is not None:
                source_codec = cam.image_codec_from_blob(_blob_to_bytes(sample))

        output_format = cam.resolve_output_format(kind=kind, source_codec=source_codec, requested=requested)
        resolved.append(
            ResolvedCamera(
                key=spec["key"], path=path, kind=kind, source_codec=source_codec, output_format=output_format
            )
        )
    return resolved


def infer_camera_feature(camera: ResolvedCamera, inference_table: pa.Table, index_column: str) -> FeatureSpec:
    """Infer the LeRobot feature spec (dtype + (H, W, 3) shape) for one camera."""
    height, width = _infer_hw(camera, inference_table, index_column)
    return {
        "dtype": camera.feature_dtype,
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }


def _infer_hw(camera: ResolvedCamera, table: pa.Table, index_column: str) -> tuple[int, int]:
    if camera.kind == "video":
        sample_bytes, times_ns = extract_first_video_sample(
            table, sample_column=camera.sample_column, time_column=index_column
        )
        frame = decode_video_frames_at_times(
            samples=[sample_bytes],
            times_ns=times_ns,
            target_times_ns=times_ns[:1],
            video_format=camera.source_codec or "h264",
        )[0]
        return int(frame.shape[0]), int(frame.shape[1])

    if camera.kind == "encoded_image":
        blob = _first_non_null(table[camera.sample_column])
        if blob is None:
            raise ValueError(f"No image samples found for camera '{camera.key}' at '{camera.path}'.")
        frame = cam.decode_encoded_image(_blob_to_bytes(blob))
        return int(frame.shape[0]), int(frame.shape[1])

    # raw_image: read width/height from the Image:format struct.
    fmt_col = f"{camera.path}:Image:format"
    fmt = _first_non_null(table[fmt_col]) if fmt_col in table.column_names else None
    if not isinstance(fmt, dict):
        raise ValueError(f"Missing Image:format for raw camera '{camera.key}' at '{camera.path}'.")
    return int(fmt["height"]), int(fmt["width"])


def _decode_raw_row(buffer_value: object, fmt_value: object) -> npt.NDArray[np.uint8]:
    fmt = fmt_value if isinstance(fmt_value, dict) else {}
    width = int(fmt["width"])
    height = int(fmt["height"])
    data = _blob_to_bytes(buffer_value)
    channels = len(data) // (width * height) if width and height else 0
    color_model = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels)
    if color_model is None:
        raise ValueError(
            f"Unsupported raw image: {len(data)} bytes for {width}x{height} (=> {channels} channels). "
            "Only 8-bit L/RGB/RGBA images are supported."
        )
    return cam.decode_raw_image(data, width=width, height=height, color_model=color_model, channels=channels)


def extract_camera_frames_at_times(
    camera: ResolvedCamera,
    table: pa.Table,
    *,
    index_column: str,
    target_times_ns: npt.NDArray[np.int64],
) -> list[npt.NDArray[np.uint8]]:
    """Decode one RGB frame per target time for a camera, using latest-at selection."""
    if camera.kind == "video":
        samples: list[bytes] = []
        times: list[object] = []
        raw_samples = table[camera.sample_column].to_pylist()
        raw_times = table[index_column].to_pylist()
        for sample, timestamp in zip(raw_samples, raw_times, strict=False):
            sample = unwrap_singleton(sample)
            if sample is None:
                continue
            samples.append(_blob_to_bytes(sample))
            times.append(timestamp)
        if not samples:
            raise ValueError(f"No video samples for camera '{camera.key}'.")
        return decode_video_frames_at_times(
            samples=samples,
            times_ns=normalize_times(times),
            target_times_ns=target_times_ns,
            video_format=camera.source_codec or "h264",
        )

    # Image sources: decode each present frame once, then map to target times (latest-at).
    frame_times: list[int] = []
    frames: list[npt.NDArray[np.uint8]] = []
    times_raw = table[index_column].to_pylist()
    samples_raw = table[camera.sample_column].to_pylist()
    formats_raw = (
        table[f"{camera.path}:Image:format"].to_pylist()
        if camera.kind == "raw_image" and f"{camera.path}:Image:format" in table.column_names
        else [None] * len(samples_raw)
    )
    for i, sample in enumerate(samples_raw):
        if sample is None:
            continue
        if camera.kind == "encoded_image":
            frames.append(cam.decode_encoded_image(_blob_to_bytes(sample)))
        else:
            frames.append(_decode_raw_row(sample, formats_raw[i]))
        frame_times.append(int(normalize_times([times_raw[i]])[0]))

    if not frames:
        raise ValueError(f"No image samples for camera '{camera.key}' at '{camera.path}'.")

    selected = _latest_at_indices(np.asarray(frame_times, dtype=np.int64), target_times_ns)
    return [frames[i] for i in selected]
