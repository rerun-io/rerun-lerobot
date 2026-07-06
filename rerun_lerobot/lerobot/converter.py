"""Main conversion logic for RRD to LeRobot dataset conversion."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datafusion as dfn
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from lerobot.datasets.compute_stats import compute_episode_stats
from tqdm import tqdm

from rerun_lerobot.lerobot.cameras import extract_camera_frames_at_times
from rerun_lerobot.lerobot.video_processing import (
    can_remux_video,
    load_video_samples,
    remux_video_stream,
)
from rerun_lerobot.utils import normalize_times, suppress_ffmpeg_output, to_float32_vector, unwrap_singleton

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rerun_lerobot.camera import ResolvedCamera
    from rerun_lerobot.lerobot.types import (
        FeatureSpec,
        LeRobotConversionConfig,
        RemuxInfo,
    )

_VIDEO_KEY_PREFIX = "observation.images."


def convert_dataframe_to_episode(
    df: dfn.DataFrame,
    config: LeRobotConversionConfig,
    *,
    lerobot_dataset: LeRobotDataset,
    segment_id: str,
    features: dict[str, FeatureSpec],
    cameras: list[ResolvedCamera],
) -> None:
    """
    Convert a DataFusion dataframe to a LeRobot episode.

    Args:
        df: DataFusion dataframe containing the segment data (already filtered and aligned)
        config: Conversion configuration
        lerobot_dataset: LeRobot dataset to add frames to
        segment_id: ID of the segment being processed (for logging)
        features: Feature specifications from inference
        cameras: Resolved cameras (source + output format) to include

    """
    action_spec = features.get("action")
    state_spec = features.get("observation.state")

    action_dim = action_spec["shape"][0] if action_spec else None
    state_dim = state_spec["shape"][0] if state_spec else None

    if action_dim is None:
        raise ValueError("Action feature specification is missing.")

    if state_dim is None:
        raise ValueError("State feature specification is missing.")

    cached_df = df.cache()

    # For video cameras whose source codec already matches the requested output,
    # load the raw packets so we can remux (copy) them instead of re-encoding.
    remux_infos: dict[str, RemuxInfo] = {}
    remux_video_format: dict[str, str] = {}
    remux_candidates = [camera for camera in cameras if camera.can_remux]
    if remux_candidates:
        packet_cache = load_video_samples(
            df=cached_df,
            index_column=config.index_column,
            videos=[{"key": camera.key, "path": camera.path} for camera in remux_candidates],
        )
        for camera in remux_candidates:
            samples, times_ns = packet_cache[camera.key]
            ok, source_fps = can_remux_video(times_ns, config.fps)
            if ok:
                remux_infos[camera.key] = {"samples": samples, "times_ns": times_ns, "source_fps": source_fps}
                remux_video_format[camera.key] = camera.source_codec or "h264"

    df = cached_df.filter(dfn.col(config.action).is_not_null())
    table = pa.table(df)
    if table.num_rows == 0:
        return

    data_columns = {name: table[name].to_pylist() for name in table.column_names}
    num_rows = table.num_rows

    # Fast path: every camera is a video we can remux (no decoding at all).
    all_remuxable = bool(cameras) and all(camera.key in remux_infos for camera in cameras)
    if all_remuxable:
        _save_episode_without_video_decode(
            lerobot_dataset=lerobot_dataset,
            data_columns=data_columns,
            num_rows=num_rows,
            config=config,
            action_dim=action_dim,
            state_dim=state_dim,
            remux_infos=remux_infos,
            remux_video_format=remux_video_format,
        )
        return

    # General path: decode every camera's frames, let LeRobot store them (PNG for
    # image-dtype cameras, encoded video for video-dtype cameras), then any remuxable
    # video is written from a lossless copy of the original packets instead of a
    # re-encode.
    #
    # Camera frames live on their own rows (their source timeline), which differ
    # from the action rows, so extract from the UNFILTERED frame and sample each
    # camera at the output row times (latest-at).
    row_times_ns = normalize_times(data_columns[config.index_column])
    full_table = pa.table(cached_df)
    camera_frames: dict[str, list[npt.NDArray[np.uint8]]] = {
        camera.key: extract_camera_frames_at_times(
            camera, full_table, index_column=config.index_column, target_times_ns=row_times_ns
        )
        for camera in cameras
    }

    for row_idx in tqdm(range(num_rows), desc=f"Frames ({segment_id})", leave=False):
        frame = _build_frame(
            row_idx=row_idx,
            data_columns=data_columns,
            config=config,
            action_dim=action_dim,
            state_dim=state_dim,
            cameras=cameras,
            camera_frames=camera_frames,
            num_rows=num_rows,
        )
        lerobot_dataset.add_frame(frame)

    if remux_infos:
        # Some cameras can be remuxed: take over the save so the remuxable videos
        # are written from the original packets while the rest are encoded normally.
        buffer = lerobot_dataset.writer.episode_buffer
        _finalize_episode(
            lerobot_dataset,
            episode_buffer=buffer,
            num_rows=int(buffer["size"]),
            tasks=list(buffer["task"]),
            remux_infos=remux_infos,
            remux_video_format=remux_video_format,
            skip_video_stats=False,
        )
    else:
        # No remuxable cameras: let LeRobot encode and store everything.
        lerobot_dataset.save_episode()


def _build_tasks(
    data_columns: dict[str, list[object]],
    config: LeRobotConversionConfig,
    num_rows: int,
) -> list[str]:
    """Resolve the per-row task string, falling back to ``config.task_default``."""
    task_col = config.task
    task_values = data_columns.get(task_col, [None] * num_rows) if task_col else [None] * num_rows

    tasks: list[str] = []
    for task_value in task_values:
        task_value = unwrap_singleton(task_value)
        if task_value is None:
            task = config.task_default
        elif isinstance(task_value, (bytes, bytearray, memoryview)):
            task = bytes(task_value).decode("utf-8")
        else:
            task = str(task_value)
        tasks.append(task)
    return tasks


def _finalize_episode(
    lerobot_dataset: LeRobotDataset,
    *,
    episode_buffer: dict[str, Any],
    num_rows: int,
    tasks: list[str],
    remux_infos: dict[str, RemuxInfo],
    remux_video_format: dict[str, str],
    skip_video_stats: bool,
) -> None:
    """
    Write one episode, remuxing the videos in ``remux_infos`` from source packets.

    This mirrors :meth:`lerobot.datasets.dataset_writer.DatasetWriter.save_episode`,
    but replaces the per-camera video encoding: any camera present in ``remux_infos``
    is written by remuxing (copying) its original compressed packets, while the rest
    are encoded normally by LeRobot. Video files are written through LeRobot's own
    :meth:`_save_episode_video`, so multi-episode chunking/concatenation stays correct.

    ``episode_buffer`` must already hold the per-feature frame lists (as produced by
    ``DatasetWriter.add_frame`` in the general path, or built by hand in the fast path).
    Set ``skip_video_stats`` when no frames were decoded to disk (fast path), so video
    statistics are not computed for the remuxed cameras.
    """
    writer = lerobot_dataset.writer
    meta = lerobot_dataset.meta
    episode_index = meta.total_episodes
    episode_tasks = list(dict.fromkeys(tasks))

    # Mirror DatasetWriter.save_episode's buffer preparation.
    episode_buffer["index"] = np.arange(meta.total_frames, meta.total_frames + num_rows, dtype=np.int64)
    episode_buffer["episode_index"] = np.full((num_rows,), episode_index, dtype=np.int64)
    meta.save_episode_tasks(episode_tasks)
    episode_buffer["task_index"] = np.array([meta.get_task_index(task) for task in tasks], dtype=np.int64)

    for key, ft in meta.features.items():
        if key in ("index", "episode_index", "task_index") or ft["dtype"] in ("image", "video"):
            continue
        stacked = np.stack(episode_buffer[key])
        # `shape=(1,)` numeric features are serialized as scalars; normalize to `(N,)`.
        if tuple(ft["shape"]) == (1,) and ft["dtype"] != "string":
            stacked = stacked.reshape(num_rows)
        episode_buffer[key] = stacked

    writer._wait_image_writer()

    if skip_video_stats:
        # No frames were decoded to disk, so image/video statistics can't be computed.
        stats_features = {key: ft for key, ft in meta.features.items() if ft["dtype"] not in ("image", "video")}
        stats_buffer = {key: episode_buffer[key] for key in stats_features}
        ep_stats = compute_episode_stats(stats_buffer, stats_features)
    else:
        ep_stats = compute_episode_stats(episode_buffer, meta.features)

    ep_metadata = writer._save_episode_data(episode_buffer)

    for video_key in meta.video_keys:
        camera_key = video_key[len(_VIDEO_KEY_PREFIX) :]
        info = remux_infos.get(camera_key)
        if info is not None:
            # Write the video by remuxing the original packets (no re-encode).
            temp_dir = Path(tempfile.mkdtemp(dir=lerobot_dataset.root))
            temp_path = temp_dir / f"{video_key.replace('.', '_')}_{episode_index:03d}.mp4"
            with suppress_ffmpeg_output():
                times_ns = info["times_ns"]
                remux_video_stream(
                    samples=info["samples"],
                    times_ns=times_ns - times_ns[0],
                    output_path=str(temp_path),
                    video_format=remux_video_format[camera_key],
                )
            ep_metadata.update(writer._save_episode_video(video_key, episode_index, temp_path=temp_path))
        else:
            # Encode from the PNG frames written by add_frame.
            ep_metadata.update(writer._save_episode_video(video_key, episode_index))

    meta.save_episode(episode_index, num_rows, episode_tasks, ep_stats, ep_metadata)
    writer.clear_episode_buffer(delete_images=len(meta.image_keys) > 0)


def _save_episode_without_video_decode(
    lerobot_dataset: LeRobotDataset,
    *,
    data_columns: dict[str, list[object]],
    num_rows: int,
    config: LeRobotConversionConfig,
    action_dim: int,
    state_dim: int,
    remux_infos: dict[str, RemuxInfo],
    remux_video_format: dict[str, str],
) -> None:
    """Save an episode without decoding video frames by remuxing source packets directly."""
    writer = lerobot_dataset.writer
    episode_index = lerobot_dataset.meta.total_episodes

    tasks = _build_tasks(data_columns, config, num_rows)

    times_ns = normalize_times(data_columns[config.index_column])
    times_s = (times_ns - times_ns[0]) / 1_000_000_000.0

    # Build the episode buffer by hand (no frames are decoded or written to disk).
    # Video keys stay empty; they are written from the remuxed packets below.
    episode_buffer = writer._create_episode_buffer(episode_index)
    episode_buffer["timestamp"] = list(times_s.astype(np.float32))
    episode_buffer["frame_index"] = list(range(num_rows))

    if action_dim is not None:
        episode_buffer["action"] = [
            to_float32_vector(value, action_dim, "action") for value in data_columns[config.action]
        ]

    if state_dim is not None:
        episode_buffer["observation.state"] = [
            to_float32_vector(value, state_dim, "state") for value in data_columns[config.state]
        ]

    _finalize_episode(
        lerobot_dataset,
        episode_buffer=episode_buffer,
        num_rows=num_rows,
        tasks=tasks,
        remux_infos=remux_infos,
        remux_video_format=remux_video_format,
        skip_video_stats=True,
    )


def _build_frame(
    *,
    row_idx: int,
    data_columns: dict[str, list[object]],
    config: LeRobotConversionConfig,
    action_dim: int,
    state_dim: int,
    cameras: list[ResolvedCamera],
    camera_frames: dict[str, list[npt.NDArray[np.uint8]]],
    num_rows: int,
) -> dict[str, object]:
    """
    Build a single frame dictionary for the LeRobot dataset.

    Args:
        row_idx: Row index in the batch
        data_columns: Dictionary of column data
        config: Conversion configuration
        action_dim: Action dimension
        state_dim: State dimension
        cameras: Resolved cameras
        camera_frames: Decoded frames per camera key
        num_rows: Total number of rows in batch

    Returns:
        Frame dictionary ready for the LeRobot dataset

    """
    frame: dict[str, object] = {}

    frame["action"] = to_float32_vector(
        data_columns[config.action][row_idx],
        action_dim,
        "action",
    )

    frame["observation.state"] = to_float32_vector(
        data_columns[config.state][row_idx],
        state_dim,
        "state",
    )

    # Add task
    task_col = config.task
    task_value = data_columns.get(task_col, [None] * num_rows)[row_idx] if task_col else None
    task_value = unwrap_singleton(task_value)
    if task_value is None:
        task = config.task_default
    elif isinstance(task_value, (bytes, bytearray, memoryview)):
        task = bytes(task_value).decode("utf-8")
    else:
        task = str(task_value)
    frame["task"] = task

    # Add camera frames (LeRobot stores each as PNG or video per its feature dtype)
    for camera in cameras:
        frame[camera.feature_key] = camera_frames[camera.key][row_idx]

    return frame
