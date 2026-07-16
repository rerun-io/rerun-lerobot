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
    extract_video_samples,
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


def _valid_indices(table: pa.Table, column: str) -> npt.NDArray[np.intp]:
    """Indices of the non-null rows of a column."""
    valid = table[column].combine_chunks().is_valid().to_numpy(zero_copy_only=False)
    return np.flatnonzero(valid)


def align_scalar_columns(
    table: pa.Table,
    config: LeRobotConversionConfig,
) -> tuple[npt.NDArray[np.int64], dict[str, list[object]]]:
    """Align every scalar column to the reference column's non-null rows via per-column latest-at."""
    times_ns = normalize_times(table[config.index_column].combine_chunks().to_numpy(zero_copy_only=False))

    # The latest-at lookups below (and the episode timestamps derived from the
    # returned times) require rows in ascending index order.
    if times_ns.size and np.any(np.diff(times_ns) < 0):
        order = np.argsort(times_ns, kind="stable")
        table = table.take(pa.array(order))
        times_ns = times_ns[order]

    reference_column = config.reference_column
    reference_rows = _valid_indices(table, reference_column)
    if reference_rows.size == 0:
        return np.array([], dtype=np.int64), {}
    row_times_ns = times_ns[reference_rows]

    scalar_columns = list(dict.fromkeys([*config.action_columns, *config.state_columns]))
    if config.task:
        scalar_columns.append(config.task)

    aligned: dict[str, list[object]] = {}
    for column in scalar_columns:
        if column == reference_column:
            take_rows: npt.NDArray[np.intp] = reference_rows
        else:
            valid_rows = _valid_indices(table, column)
            if valid_rows.size == 0:
                if column == config.task:
                    aligned[column] = [None] * len(row_times_ns)
                    continue
                raise ValueError(f"Column '{column}' has no values in this segment.")
            positions = np.searchsorted(times_ns[valid_rows], row_times_ns, side="right") - 1
            take_rows = valid_rows[np.clip(positions, 0, valid_rows.size - 1)]
        aligned[column] = table[column].take(pa.array(take_rows)).to_pylist()
    return row_times_ns, aligned


def _feature_vectors(
    aligned: dict[str, list[object]],
    columns: list[str],
    expected_dim: int,
    label: str,
) -> list[npt.NDArray[np.float32]]:
    """One float32 vector per output row, concatenating the columns' aligned values in order."""
    num_rows = len(aligned[columns[0]])
    vectors: list[npt.NDArray[np.float32]] = []
    for row_idx in range(num_rows):
        if len(columns) == 1:
            value: object = aligned[columns[0]][row_idx]
        else:
            parts = [
                np.asarray(unwrap_singleton(aligned[column][row_idx]), dtype=np.float32).flatten() for column in columns
            ]
            value = np.concatenate(parts)
        vectors.append(to_float32_vector(value, expected_dim, label))
    return vectors


def convert_dataframe_to_episode(
    df: dfn.DataFrame,
    config: LeRobotConversionConfig,
    *,
    lerobot_dataset: LeRobotDataset,
    segment_id: str,
    features: dict[str, FeatureSpec],
    cameras: list[ResolvedCamera],
    camera_dfs: dict[str, dfn.DataFrame],
) -> None:
    """
    Convert a DataFusion dataframe to a LeRobot episode.

    Args:
        df: DataFusion dataframe with the segment's scalar data (already filtered and indexed)
        config: Conversion configuration
        lerobot_dataset: LeRobot dataset to add frames to
        segment_id: ID of the segment being processed (for logging)
        features: Feature specifications from inference
        cameras: Resolved cameras (source + output format) to include
        camera_dfs: Per-camera dataframe (keyed by camera key), on the camera's own index

    """
    action_spec = features.get("action")
    state_spec = features.get("observation.state")

    action_dim = action_spec["shape"][0] if action_spec else None
    state_dim = state_spec["shape"][0] if state_spec else None

    if action_dim is None:
        raise ValueError("Action feature specification is missing.")

    if state_dim is None:
        raise ValueError("State feature specification is missing.")

    table = pa.table(df)
    if table.num_rows == 0:
        return

    row_times_ns, aligned = align_scalar_columns(table, config)
    num_rows = len(row_times_ns)
    if num_rows == 0:
        return

    actions = _feature_vectors(aligned, config.action_columns, action_dim, "action")
    states = _feature_vectors(aligned, config.state_columns, state_dim, "state")
    tasks = _build_tasks(aligned.get(config.task) if config.task else None, config, num_rows)

    # For video cameras whose source codec already matches the requested output,
    # load the raw packets so we can remux (copy) them instead of re-encoding.
    remux_infos: dict[str, RemuxInfo] = {}
    remux_video_format: dict[str, str] = {}
    for camera in cameras:
        if not camera.can_remux:
            continue
        samples, times_ns = _load_camera_video_samples(camera_dfs[camera.key], camera)
        ok, source_fps = can_remux_video(times_ns, config.fps)
        if ok:
            remux_infos[camera.key] = {"samples": samples, "times_ns": times_ns, "source_fps": source_fps}
            remux_video_format[camera.key] = camera.source_codec or "h264"

    # Fast path: every camera is a video we can remux (no decoding at all).
    all_remuxable = bool(cameras) and all(camera.key in remux_infos for camera in cameras)
    if all_remuxable:
        _save_episode_without_video_decode(
            lerobot_dataset=lerobot_dataset,
            row_times_ns=row_times_ns,
            actions=actions,
            states=states,
            tasks=tasks,
            remux_infos=remux_infos,
            remux_video_format=remux_video_format,
        )
        return

    # General path: decode every camera's frames, let LeRobot store them (PNG for
    # image-dtype cameras, encoded video for video-dtype cameras), then any remuxable
    # video is written from a lossless copy of the original packets instead of a
    # re-encode.
    #
    # Each camera is sampled from its own dataframe at the output row times
    # (latest-at); cameras on a different timeline align by elapsed time.
    camera_frames: dict[str, list[npt.NDArray[np.uint8]]] = {}
    for camera in cameras:
        camera_table = pa.table(camera_dfs[camera.key])
        elapsed_alignment = camera.index_column != config.index_column
        camera_frames[camera.key] = extract_camera_frames_at_times(
            camera,
            camera_table,
            index_column=camera.index_column,
            target_times_ns=row_times_ns - row_times_ns[0] if elapsed_alignment else row_times_ns,
            elapsed_alignment=elapsed_alignment,
        )

    for row_idx in tqdm(range(num_rows), desc=f"Frames ({segment_id})", leave=False):
        frame: dict[str, object] = {
            "action": actions[row_idx],
            "observation.state": states[row_idx],
            "task": tasks[row_idx],
        }
        for camera in cameras:
            frame[camera.feature_key] = camera_frames[camera.key][row_idx]
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


def _load_camera_video_samples(
    camera_df: dfn.DataFrame,
    camera: ResolvedCamera,
) -> tuple[list[bytes], npt.NDArray[np.int64]]:
    """Load one camera's raw video packets and timestamps from its own dataframe."""
    # No .select(): projecting by name silently drops rows for duration-typed indexes.
    video_table = pa.table(camera_df.filter(dfn.col(camera.sample_column).is_not_null()))
    return extract_video_samples(
        video_table,
        sample_column=camera.sample_column,
        time_column=camera.index_column,
    )


def _build_tasks(
    task_values: list[object] | None,
    config: LeRobotConversionConfig,
    num_rows: int,
) -> list[str]:
    """Resolve the per-row task string, falling back to ``config.task_default``."""
    if task_values is None:
        task_values = [None] * num_rows

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
    row_times_ns: npt.NDArray[np.int64],
    actions: list[npt.NDArray[np.float32]],
    states: list[npt.NDArray[np.float32]],
    tasks: list[str],
    remux_infos: dict[str, RemuxInfo],
    remux_video_format: dict[str, str],
) -> None:
    """Save an episode without decoding video frames by remuxing source packets directly."""
    writer = lerobot_dataset.writer
    episode_index = lerobot_dataset.meta.total_episodes
    num_rows = len(row_times_ns)

    times_s = (row_times_ns - row_times_ns[0]) / 1_000_000_000.0

    # Build the episode buffer by hand (no frames are decoded or written to disk).
    # Video keys stay empty; they are written from the remuxed packets below.
    episode_buffer = writer._create_episode_buffer(episode_index)
    episode_buffer["timestamp"] = list(times_s.astype(np.float32))
    episode_buffer["frame_index"] = list(range(num_rows))
    episode_buffer["action"] = actions
    episode_buffer["observation.state"] = states

    _finalize_episode(
        lerobot_dataset,
        episode_buffer=episode_buffer,
        num_rows=num_rows,
        tasks=tasks,
        remux_infos=remux_infos,
        remux_video_format=remux_video_format,
        skip_video_stats=True,
    )
