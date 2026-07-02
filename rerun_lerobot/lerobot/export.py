"""High-level dataset export: RRD directory or remote Rerun catalog to LeRobot v3."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import rerun as rr
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from rerun_lerobot.lerobot.converter import convert_dataframe_to_episode
from rerun_lerobot.lerobot.feature_inference import infer_features

if TYPE_CHECKING:
    from rerun.catalog import DatasetEntry

    from rerun_lerobot.lerobot.types import LeRobotConversionConfig


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

    # Build list of all columns needed for feature inference
    inference_columns = [config.index_column, config.action, config.state]
    if config.task:
        inference_columns.append(config.task)
    for spec in config.videos:
        inference_columns.append(f"{spec['path']}:VideoStream:sample")

    # Query all columns from one segment
    inference_view = dataset.filter_segments(inference_segment_id).filter_contents(contents)
    inference_reader = inference_view.reader(index=config.index_column)
    inference_table = pa.table(inference_reader.select(*inference_columns))

    print("Inferring features from segment:", inference_segment_id)
    start_time = time.time()
    features = infer_features(
        table=inference_table,
        config=config,
    )
    end_time = time.time()
    print(f"Inferring features took {end_time - start_time:.2f} seconds")

    # Create LeRobot dataset
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=config.fps,
        features=features,
        root=output_dir,
        use_videos=config.use_videos,
    )

    # Process each segment
    for segment_id in tqdm(segment_ids, desc="Segments"):
        try:
            contents, reference_path = config.get_filter_list()

            if reference_path is None:
                print(f"Skipping segment '{segment_id}': no action or state column specified")
                continue

            # Check if segment is empty
            segment_table = dataset.segment_table()
            segment_info = pa.table(segment_table.df)
            is_empty = False
            for i in range(segment_info.num_rows):
                if segment_info["rerun_segment_id"][i].as_py() == segment_id:
                    size_bytes = segment_info["rerun_size_bytes"][i].as_py()
                    if size_bytes == 0:
                        print(f"Skipping segment '{segment_id}': segment is empty (0 bytes)")
                        is_empty = True
                        break
            if is_empty:
                continue

            view = dataset.filter_segments(segment_id).filter_contents(contents)
            df = view.reader(
                index=config.index_column,
            )

            convert_dataframe_to_episode(
                df,
                config,
                lerobot_dataset=lerobot_dataset,
                segment_id=segment_id,
                features=features,
            )

        except Exception as err:
            print(f"Error processing segment {segment_id}: {err}")
            import traceback

            traceback.print_exc()
            continue

    lerobot_dataset.finalize()


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
        catalog_url: URL of the Rerun catalog server (e.g. ``rerun+http://host:port``).
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
