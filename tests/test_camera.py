"""Tests for camera source detection, output-format resolution, and image decoding."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image as PILImage

from rerun_lerobot.camera import (
    decode_encoded_image,
    decode_raw_image,
    detect_camera_kind,
    image_codec_from_blob,
    resolve_output_format,
    validate_output_format,
)


def test_detect_camera_kind() -> None:
    names = [
        "/cam/vid:VideoStream:sample",
        "/cam/jpg:EncodedImage:blob",
        "/cam/raw:Image:buffer",
        "/other:Scalars:scalars",
    ]
    assert detect_camera_kind(names, "/cam/vid") == "video"
    assert detect_camera_kind(names, "/cam/jpg") == "encoded_image"
    assert detect_camera_kind(names, "/cam/raw") == "raw_image"


def test_detect_camera_kind_unsupported() -> None:
    with pytest.raises(ValueError, match="No supported camera archetype"):
        detect_camera_kind(["/x:Scalars:scalars"], "/x")


def test_validate_output_format() -> None:
    assert validate_output_format(None) is None
    assert validate_output_format("PNG") == "png"
    assert validate_output_format("h264") == "h264"
    with pytest.raises(ValueError, match="jpg.*not supported|PNG only"):
        validate_output_format("jpg")
    with pytest.raises(ValueError, match="Invalid --output-format"):
        validate_output_format("gif")


def test_resolve_output_format_explicit_wins() -> None:
    assert resolve_output_format(kind="video", source_codec="h264", requested="av1") == "av1"
    assert resolve_output_format(kind="raw_image", source_codec=None, requested="png") == "png"


def test_resolve_output_format_keep_original() -> None:
    # video: keep the codec if storable, else h264
    assert resolve_output_format(kind="video", source_codec="h264", requested=None) == "h264"
    assert resolve_output_format(kind="video", source_codec="hevc", requested=None) == "hevc"
    assert resolve_output_format(kind="video", source_codec="av1", requested=None) == "av1"
    assert resolve_output_format(kind="video", source_codec="mpeg4", requested=None) == "h264"
    # encoded image: png stays png, jpeg -> h264
    assert resolve_output_format(kind="encoded_image", source_codec="png", requested=None) == "png"
    assert resolve_output_format(kind="encoded_image", source_codec="jpeg", requested=None) == "h264"
    # raw -> h264
    assert resolve_output_format(kind="raw_image", source_codec=None, requested=None) == "h264"


def test_image_codec_and_decode_roundtrip() -> None:
    arr = np.dstack([
        np.full((8, 12), 200, np.uint8),
        np.full((8, 12), 100, np.uint8),
        np.full((8, 12), 50, np.uint8),
    ])
    for pil_fmt, expected in [("PNG", "png"), ("JPEG", "jpeg")]:
        buf = io.BytesIO()
        PILImage.fromarray(arr).save(buf, format=pil_fmt)
        blob = buf.getvalue()
        assert image_codec_from_blob(blob) == expected
        decoded = decode_encoded_image(blob)
        assert decoded.shape == (8, 12, 3)
        assert decoded.dtype == np.uint8
        if pil_fmt == "PNG":  # lossless
            np.testing.assert_array_equal(decoded, arr)


def test_decode_raw_image_rgb() -> None:
    arr = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    out = decode_raw_image(arr.tobytes(), width=5, height=4, color_model="RGB", channels=3)
    assert out.shape == (4, 5, 3)
    np.testing.assert_array_equal(out, arr)


def test_decode_raw_image_grayscale_expands_to_rgb() -> None:
    arr = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5, 1)
    out = decode_raw_image(arr.tobytes(), width=5, height=4, color_model="L", channels=1)
    assert out.shape == (4, 5, 3)
    np.testing.assert_array_equal(out[:, :, 0], arr[:, :, 0])
    np.testing.assert_array_equal(out[:, :, 0], out[:, :, 2])


def test_decode_raw_image_bgr_swaps() -> None:
    bgr = np.dstack([
        np.full((2, 2), 1, np.uint8),
        np.full((2, 2), 2, np.uint8),
        np.full((2, 2), 3, np.uint8),
    ])
    out = decode_raw_image(bgr.tobytes(), width=2, height=2, color_model="BGR", channels=3)
    assert out[0, 0, 0] == 3 and out[0, 0, 2] == 1  # channels reversed to RGB


def test_decode_raw_image_size_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        decode_raw_image(b"\x00\x01\x02", width=5, height=4, color_model="RGB", channels=3)
