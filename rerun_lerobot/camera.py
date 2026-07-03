"""
Camera source detection and output-format resolution.

A camera entity in a Rerun recording can store frames in several ways:

- ``VideoStream``  — compressed video packets (H.264 / HEVC / AV1).
- ``EncodedImage`` — per-frame encoded images (JPEG / PNG).
- ``Image``        — raw pixel buffers.

LeRobot can only *store* two things: PNG image frames (``dtype: "image"``) or an
MP4 video (``dtype: "video"``) encoded with H.264 / HEVC / AV1. The user picks
the output with ``--output-format {png,h264,hevc,av1}``; when omitted we keep the
source format if LeRobot can store it, otherwise fall back to H.264.

This module has no heavy dependencies beyond Pillow/NumPy so it stays unit-testable.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage

# Output formats we can hand to LeRobot.
OUTPUT_IMAGE_FORMATS = ("png",)
OUTPUT_VIDEO_FORMATS = ("h264", "hevc", "av1")
OUTPUT_FORMATS = (*OUTPUT_IMAGE_FORMATS, *OUTPUT_VIDEO_FORMATS)

# Our output-format name -> the vcodec string LeRobotDataset.create expects.
LEROBOT_VCODEC = {"h264": "h264", "hevc": "hevc", "av1": "libsvtav1"}

# Video codecs LeRobot can store, so a matching source can be remuxed (kept) as-is.
KEEPABLE_VIDEO_CODECS = frozenset({"h264", "hevc", "av1"})

_KIND_VIDEO = "video"
_KIND_ENCODED_IMAGE = "encoded_image"
_KIND_RAW_IMAGE = "raw_image"


@dataclass(frozen=True)
class CameraSource:
    """How a camera entity stores its frames in the recording."""

    key: str
    path: str
    kind: str  # _KIND_VIDEO | _KIND_ENCODED_IMAGE | _KIND_RAW_IMAGE
    # For video: 'h264'/'hevc'/'av1' (or None if not yet probed).
    # For encoded_image: 'jpeg'/'png'. For raw_image: None.
    source_codec: str | None = None

    @property
    def is_video(self) -> bool:
        return self.kind == _KIND_VIDEO

    @property
    def sample_column(self) -> str:
        """The primary data column for this source."""
        if self.kind == _KIND_VIDEO:
            return f"{self.path}:VideoStream:sample"
        if self.kind == _KIND_ENCODED_IMAGE:
            return f"{self.path}:EncodedImage:blob"
        return f"{self.path}:Image:buffer"


def detect_camera_kind(schema_names: list[str], path: str) -> str:
    """
    Detect how a camera entity stores frames, from the dataset's column names.

    Raises:
        ValueError: If the path has no supported camera archetype, with guidance.

    """
    if f"{path}:VideoStream:sample" in schema_names:
        return _KIND_VIDEO
    if f"{path}:EncodedImage:blob" in schema_names:
        return _KIND_ENCODED_IMAGE
    if f"{path}:Image:buffer" in schema_names:
        return _KIND_RAW_IMAGE
    raise ValueError(
        f"No supported camera archetype found at '{path}'. Expected one of: "
        f"VideoStream (compressed video), EncodedImage (JPEG/PNG), or Image (raw pixels). "
        f"Check the entity path (see `--inspect`), or that this entity actually holds camera frames."
    )


def validate_output_format(requested: str | None) -> str | None:
    """Validate a requested --output-format, with a helpful message for jpg."""
    if requested is None:
        return None
    fmt = requested.lower()
    if fmt in ("jpg", "jpeg"):
        raise ValueError(
            "Output format 'jpg' is not supported: LeRobot stores per-frame images as PNG only. "
            "Use '--output-format png' for images, or a video codec (h264, hevc, av1)."
        )
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"Invalid --output-format '{requested}'. Choose one of: {', '.join(OUTPUT_FORMATS)}.")
    return fmt


def resolve_output_format(*, kind: str, source_codec: str | None, requested: str | None) -> str:
    """
    Decide the LeRobot output format for one camera.

    With an explicit ``requested`` format, use it. Otherwise keep the source format
    when LeRobot can store it, else fall back to H.264:

    - video h264/hevc/av1 -> same codec (remuxed, no re-encode)
    - EncodedImage PNG     -> png
    - EncodedImage JPEG    -> h264 (LeRobot can't store jpeg)
    - raw Image / anything else -> h264
    """
    if requested is not None:
        return requested

    if kind == _KIND_VIDEO:
        return source_codec if source_codec in KEEPABLE_VIDEO_CODECS else "h264"
    if kind == _KIND_ENCODED_IMAGE:
        return "png" if source_codec == "png" else "h264"
    return "h264"


def output_is_image(output_format: str) -> bool:
    return output_format in OUTPUT_IMAGE_FORMATS


def image_codec_from_blob(blob: bytes) -> str:
    """Sniff the codec of an EncodedImage blob: 'jpeg', 'png', or the lowercased PIL format."""
    fmt = PILImage.open(io.BytesIO(blob)).format
    if fmt is None:
        return "unknown"
    return {"JPEG": "jpeg", "PNG": "png"}.get(fmt, fmt.lower())


def decode_encoded_image(blob: bytes) -> npt.NDArray[np.uint8]:
    """Decode a JPEG/PNG EncodedImage blob to an (H, W, 3) uint8 RGB array."""
    image = PILImage.open(io.BytesIO(blob)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


# Rerun `Image:format` color_model enum -> channel count. Only 8-bit unsigned,
# non-planar formats are supported; anything else raises with guidance.
_COLOR_MODEL_CHANNELS = {"L": 1, "RGB": 3, "RGBA": 4, "BGR": 3, "BGRA": 4}
_BGR_MODELS = {"BGR", "BGRA"}


def decode_raw_image(buffer: bytes, *, width: int, height: int, color_model: str, channels: int) -> npt.NDArray[np.uint8]:
    """
    Decode a raw ``Image`` buffer (8-bit) to an (H, W, 3) uint8 RGB array.

    Args:
        buffer: Raw pixel bytes.
        width: Image width in pixels.
        height: Image height in pixels.
        color_model: Color model name (e.g. 'RGB', 'RGBA', 'L', 'BGR').
        channels: Number of channels implied by the color model.

    Raises:
        ValueError: If the buffer size doesn't match, or the format is unsupported.

    """
    expected = width * height * channels
    if len(buffer) != expected:
        raise ValueError(
            f"Raw image buffer size {len(buffer)} does not match {width}x{height}x{channels}={expected}. "
            "Only 8-bit images are supported; higher bit depths are not."
        )
    array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, channels)

    if color_model in _BGR_MODELS:
        array = array[:, :, ::-1][:, :, :3] if channels >= 3 else array
        array = np.ascontiguousarray(array[:, :, :3])
        return array.astype(np.uint8)
    if channels == 1:
        return np.ascontiguousarray(np.repeat(array, 3, axis=2)).astype(np.uint8)
    return np.ascontiguousarray(array[:, :, :3]).astype(np.uint8)
