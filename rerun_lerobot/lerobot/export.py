"""High-level dataset export: RRD directory or remote Rerun catalog to LeRobot v3."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import rerun as rr
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from rerun_lerobot.camera import LEROBOT_VCODEC
from rerun_lerobot.inspection import DatasetInspection, classify_schema
from rerun_lerobot.lerobot.cameras import resolve_cameras
from rerun_lerobot.lerobot.converter import convert_dataframe_to_episode
from rerun_lerobot.lerobot.feature_inference import infer_features
from rerun_lerobot.utils import split_dataset_url

if TYPE_CHECKING:
    import datafusion as dfn
    from rerun.catalog import DatasetEntry

    from rerun_lerobot.lerobot.types import LeRobotConversionConfig, VideoSpec


def _dataset_vcodec(output_format: str | None) -> str:
    """LeRobot dataset vcodec: the requested video codec, or h264 for re-encodes by default."""
    if output_format in LEROBOT_VCODEC:
        return LEROBOT_VCODEC[output_format]
    return "h264"


def _num_segments(dataset: DatasetEntry) -> int | None:
    try:
        return len(dataset.segment_ids())
    except Exception:
        return None


def _camera_reader(
    dataset: DatasetEntry,
    segment_id: str,
    config: LeRobotConversionConfig,
    spec: VideoSpec,
) -> dfn.DataFrame:
    """One camera's per-segment dataframe: its entity only, on its own index timeline."""
    view = dataset.filter_segments(segment_id).filter_contents([spec["path"]])
    return view.reader(index=config.camera_index_column(spec))


def inspect_dataset(dataset: DatasetEntry) -> DatasetInspection:
    """Inspect a connected dataset and classify its columns (see :mod:`rerun_lerobot.inspection`)."""
    return classify_schema(
        dataset.arrow_schema(),
        dataset_name=dataset.name,
        num_segments=_num_segments(dataset),
    )


def inspect_rrd_dataset(rrd_dir: Path, *, dataset_name: str = "rrd_dataset") -> DatasetInspection:
    """Inspect a local directory of RRD recordings, served by a local OSS server."""
    if not rrd_dir.is_dir():
        raise ValueError(f"RRD directory does not exist or is not a directory: {rrd_dir}")
    with rr.server.Server(datasets={dataset_name: rrd_dir}) as server:
        dataset = server.client().get_dataset(name=dataset_name)
        return inspect_dataset(dataset)


def inspect_catalog_dataset(*, catalog_url: str, dataset_name: str, token: str | None = None) -> DatasetInspection:
    """Inspect a dataset in a remote Rerun catalog, looked up by name."""
    client = rr.catalog.CatalogClient(catalog_url, token=token)
    return inspect_dataset(client.get_dataset(name=dataset_name))


def inspect_dataset_url(dataset_url: str, *, token: str | None = None) -> DatasetInspection:
    """Inspect a dataset addressed by a full Rerun dataset URL."""
    catalog_url, entry_id = split_dataset_url(dataset_url)
    client = rr.catalog.CatalogClient(catalog_url, token=token)
    return inspect_dataset(client.get_dataset(id=entry_id))


def convert_dataset_to_lerobot(
    dataset: DatasetEntry,
    *,
    output_dir: Path,
    repo_id: str,
    config: LeRobotConversionConfig,
    segments: list[str] | None = None,
    max_segments: int | None = None,
) -> None:
    """
    Convert an already-connected Rerun catalog dataset to a LeRobot v3 dataset.

    This is the shared core used by both :func:`convert_rrd_dataset_to_lerobot`
    (which serves a local directory of RRD files) and
    :func:`convert_catalog_dataset_to_lerobot` (which talks to a remote catalog).

    This function handles:
    1. Feature inference from a representative segment
    2. Querying dataframes for each segment
    3. Calling the conversion function with dataframes

    Args:
        dataset: A connected Rerun catalog dataset entry.
        output_dir: Output directory for the LeRobot dataset.
        repo_id: LeRobot repo ID.
        config: Conversion configuration.
        segments: Optional list of segment IDs to convert (defaults to all).
        max_segments: Optional limit on the number of segments.

    """
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")

    segment_ids = list(segments) if segments else dataset.segment_ids()
    if max_segments is not None:
        segment_ids = segment_ids[:max_segments]
    if not segment_ids:
        raise ValueError("No segments found in the dataset.")

    # Query a representative segment for feature inference
    inference_segment_id = segment_ids[0]
    contents, reference_path = config.get_filter_list()
    if reference_path is None:
        raise ValueError("No action or state column specified.")

    schema_names = list(dataset.arrow_schema().names)

    inference_columns = list(dict.fromkeys([config.index_column, *config.action_columns, *config.state_columns]))
    if config.task:
        inference_columns.append(config.task)

    # Query the scalar columns from one segment; cameras each get their own query.
    inference_view = dataset.filter_segments(inference_segment_id).filter_contents(contents)
    inference_reader = inference_view.reader(index=config.index_column)
    inference_table = pa.table(inference_reader.select(*inference_columns))

    camera_tables = {
        spec["key"]: pa.table(_camera_reader(dataset, inference_segment_id, config, spec)) for spec in config.videos
    }

    # Resolve each camera's source codec and output format.
    cameras = resolve_cameras(
        videos=config.videos,
        schema_names=schema_names,
        camera_tables=camera_tables,
        requested_format=config.output_format,
        default_index_column=config.index_column,
    )
    for camera in cameras:
        tqdm.write(f"Camera '{camera.key}': {camera.kind} ({camera.source_codec}) -> {camera.output_format}")

    tqdm.write(f"Inferring features from segment: {inference_segment_id}")
    start_time = time.time()
    features = infer_features(table=inference_table, config=config, cameras=cameras, camera_tables=camera_tables)
    end_time = time.time()
    tqdm.write(f"Inferring features took {end_time - start_time:.2f} seconds")

    # Create LeRobot dataset. use_videos if any camera outputs video; the RGB
    # encoder's vcodec is used by LeRobot when it (re-)encodes video (remuxed
    # cameras bypass it).
    use_videos = any(not camera.outputs_image for camera in cameras)
    vcodec = _dataset_vcodec(config.output_format)
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=config.fps,
        features=features,
        root=output_dir,
        use_videos=use_videos,
        **({"rgb_encoder": RGBEncoderConfig(vcodec=vcodec)} if use_videos else {}),
    )

    # Fetch segment sizes once (used to weight the progress bar and to skip empty
    # segments). Doing it once avoids an O(num_segments) round-trip in the loop.
    segment_sizes = _segment_sizes(dataset)
    total_bytes = sum(segment_sizes.get(seg, 0) for seg in segment_ids)
    use_byte_progress = total_bytes > 0

    # A single dataset-wide progress bar with an ETA. Weighted by segment size in
    # bytes when available (segments vary a lot in length), else by segment count.
    progress = tqdm(
        total=total_bytes if use_byte_progress else len(segment_ids),
        unit="B" if use_byte_progress else "seg",
        unit_scale=use_byte_progress,
        desc="Converting",
    )
    converted = 0
    skipped = 0
    with progress:
        for segment_id in segment_ids:
            step = segment_sizes.get(segment_id, 0) if use_byte_progress else 1
            try:
                if segment_sizes.get(segment_id) == 0:
                    tqdm.write(f"Skipping segment '{segment_id}': segment is empty (0 bytes)")
                    skipped += 1
                    continue

                view = dataset.filter_segments(segment_id).filter_contents(contents)
                df = view.reader(index=config.index_column)
                camera_dfs = {spec["key"]: _camera_reader(dataset, segment_id, config, spec) for spec in config.videos}

                convert_dataframe_to_episode(
                    df,
                    config,
                    lerobot_dataset=lerobot_dataset,
                    segment_id=segment_id,
                    features=features,
                    cameras=cameras,
                    camera_dfs=camera_dfs,
                )
                converted += 1

            except Exception as err:
                tqdm.write(f"Error processing segment {segment_id}: {err}")
                import traceback

                traceback.print_exc()
                skipped += 1
                continue
            finally:
                progress.update(step)
                progress.set_postfix(
                    episodes=converted,
                    frames=lerobot_dataset.meta.total_frames,
                    skipped=skipped,
                )

    lerobot_dataset.finalize()


def _segment_sizes(dataset: DatasetEntry) -> dict[str, int]:
    """Map each segment id to its size in bytes (empty dict if the metadata is unavailable)."""
    try:
        segment_info = pa.table(dataset.segment_table().df)
    except Exception:
        return {}
    if "rerun_segment_id" not in segment_info.column_names or "rerun_size_bytes" not in segment_info.column_names:
        return {}
    ids = segment_info["rerun_segment_id"].to_pylist()
    sizes = segment_info["rerun_size_bytes"].to_pylist()
    return {seg_id: int(size or 0) for seg_id, size in zip(ids, sizes, strict=False)}


def convert_rrd_dataset_to_lerobot(
    *,
    rrd_dir: Path,
    output_dir: Path,
    dataset_name: str,
    repo_id: str,
    config: LeRobotConversionConfig,
    segments: list[str] | None = None,
    max_segments: int | None = None,
) -> None:
    """
    Convert a directory of RRD recordings to a LeRobot v3 dataset.

    Serves the directory with a local OSS Rerun server and converts it.

    Args:
        rrd_dir: Directory containing RRD recordings.
        output_dir: Output directory for the LeRobot dataset.
        dataset_name: Catalog dataset name to serve the directory under.
        repo_id: LeRobot repo ID.
        config: Conversion configuration.
        segments: Optional list of segment IDs to convert (defaults to all).
        max_segments: Optional limit on the number of segments.

    """
    if not rrd_dir.is_dir():
        raise ValueError(f"RRD directory does not exist or is not a directory: {rrd_dir}")
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")

    with rr.server.Server(datasets={dataset_name: rrd_dir}) as server:
        client = server.client()
        dataset = client.get_dataset(name=dataset_name)
        convert_dataset_to_lerobot(
            dataset,
            output_dir=output_dir,
            repo_id=repo_id,
            config=config,
            segments=segments,
            max_segments=max_segments,
        )


def convert_catalog_dataset_to_lerobot(
    *,
    catalog_url: str,
    dataset_name: str,
    output_dir: Path,
    repo_id: str,
    config: LeRobotConversionConfig,
    token: str | None = None,
    segments: list[str] | None = None,
    max_segments: int | None = None,
) -> None:
    """
    Convert a dataset from a remote Rerun catalog to a LeRobot v3 dataset.

    Connects to the catalog server with :class:`rerun.catalog.CatalogClient` and
    looks the dataset up by name.

    Args:
        catalog_url: URL of the Rerun catalog server (e.g. ``rerun+http://<host>:<port>``).
        dataset_name: Name of the dataset in the catalog.
        output_dir: Output directory for the LeRobot dataset.
        repo_id: LeRobot repo ID.
        config: Conversion configuration.
        token: Optional authentication token for the catalog server.
        segments: Optional list of segment IDs to convert (defaults to all).
        max_segments: Optional limit on the number of segments.

    """
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")

    client = rr.catalog.CatalogClient(catalog_url, token=token)
    dataset = client.get_dataset(name=dataset_name)
    convert_dataset_to_lerobot(
        dataset,
        output_dir=output_dir,
        repo_id=repo_id,
        config=config,
        segments=segments,
        max_segments=max_segments,
    )


def convert_dataset_url_to_lerobot(
    *,
    dataset_url: str,
    output_dir: Path,
    repo_id: str,
    config: LeRobotConversionConfig,
    token: str | None = None,
    segments: list[str] | None = None,
    max_segments: int | None = None,
) -> None:
    """
    Convert a dataset addressed by a full Rerun dataset URL to a LeRobot v3 dataset.

    The URL bundles the catalog server and the dataset entry id, e.g.
    ``rerun://hostname:443/entry/18B40C6FA7631F942c0e90030ac230fa``.
    It is split into a catalog server URL and an entry id, and the dataset is
    looked up by id via :class:`rerun.catalog.CatalogClient`.

    Args:
        dataset_url: Full Rerun dataset entry URL.
        output_dir: Output directory for the LeRobot dataset.
        repo_id: LeRobot repo ID.
        config: Conversion configuration.
        token: Optional authentication token for the catalog server.
        segments: Optional list of segment IDs to convert (defaults to all).
        max_segments: Optional limit on the number of segments.

    """
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")

    catalog_url, entry_id = split_dataset_url(dataset_url)
    client = rr.catalog.CatalogClient(catalog_url, token=token)
    dataset = client.get_dataset(id=entry_id)
    convert_dataset_to_lerobot(
        dataset,
        output_dir=output_dir,
        repo_id=repo_id,
        config=config,
        segments=segments,
        max_segments=max_segments,
    )
