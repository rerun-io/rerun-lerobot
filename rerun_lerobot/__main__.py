#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rerun_lerobot.inspection import DatasetInspection, suggest_command
from rerun_lerobot.lerobot.export import (
    convert_catalog_dataset_to_lerobot,
    convert_dataset_to_lerobot,
    convert_dataset_url_to_lerobot,
    convert_rrd_dataset_to_lerobot,
    inspect_catalog_dataset,
    inspect_dataset_url,
    inspect_rrd_dataset,
)
from rerun_lerobot.lerobot.types import LeRobotConversionConfig, VideoSpec

__all__ = [
    "convert_catalog_dataset_to_lerobot",
    "convert_dataset_to_lerobot",
    "convert_dataset_url_to_lerobot",
    "convert_rrd_dataset_to_lerobot",
    "main",
]


def _parse_video_specs(raw_specs: list[str]) -> list[VideoSpec]:
    specs: list[VideoSpec] = []
    for raw_spec in raw_specs:
        parts = raw_spec.split(":")
        if len(parts) != 2:
            raise ValueError("Video spec must be formatted as key:path (videostream only).")
        key, path = parts
        specs.append(VideoSpec(key=key, path=path))
    return specs


def _parse_name_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return names or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Rerun recordings into a LeRobot v3 dataset, "
        "from a local directory of RRD files or a remote Rerun catalog.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rrd-dir", type=Path, help="Directory containing RRD recordings.")
    source.add_argument(
        "--catalog-url",
        default=None,
        help="URL of a Rerun catalog server (e.g. 'rerun+http://host:port'). Use with --dataset-name.",
    )
    source.add_argument(
        "--dataset-url",
        default=None,
        help="Full Rerun dataset entry URL, e.g. "
        "'rerun://hostname:443/entry/18B40C6FA7631F942c0e90030ac230fa'. "
        "Bundles the catalog server and dataset id; no --dataset-name needed.",
    )

    parser.add_argument(
        "--catalog-token",
        default=None,
        help="Optional auth token for the catalog server (use with --catalog-url or --dataset-url).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory for the LeRobot dataset.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name. Required with --catalog-url; defaults to 'rrd_dataset' with --rrd-dir.",
    )
    parser.add_argument("--repo-id", default=None, help="LeRobot repo id (defaults to dataset name).")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect the dataset and print convertible columns (candidates for --action/--state/etc), then exit.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Target dataset FPS.")
    parser.add_argument("--index", default="real_time", help="Timeline to align on (e.g. real_time).")
    parser.add_argument("--action", default=None, help="Fully qualified action column (e.g. 'path:Component:field').")
    parser.add_argument("--state", default=None, help="Fully qualified state column (e.g. 'path:Component:field').")
    parser.add_argument("--task", default=None, help="Fully qualified task column (e.g. 'path:Component:field').")
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Video stream spec as key:path. Repeatable.",
    )
    parser.add_argument("--segments", nargs="*", default=None, help="Optional list of segment ids to convert.")
    parser.add_argument("--max-segments", type=int, default=None, help="Limit number of segments.")
    parser.add_argument("--use-images", action="store_true", help="Store images inline instead of videos.")
    parser.add_argument("--action-names", default=None, help="Comma-separated action names.")
    parser.add_argument("--state-names", default=None, help="Comma-separated state names.")
    return parser.parse_args()


def _inspect_source(args: argparse.Namespace) -> tuple[DatasetInspection, list[str]]:
    """
    Inspect whichever source was selected.

    Returns the inspection plus the source CLI flags (used to build a
    copy-pasteable suggested command).
    """
    if args.dataset_url is not None:
        source_args = ["--dataset-url", args.dataset_url]
        return inspect_dataset_url(args.dataset_url, token=args.catalog_token), source_args
    if args.catalog_url is not None:
        if not args.dataset_name:
            raise ValueError("--dataset-name is required when using --catalog-url.")
        source_args = ["--catalog-url", args.catalog_url, "--dataset-name", args.dataset_name]
        inspection = inspect_catalog_dataset(
            catalog_url=args.catalog_url, dataset_name=args.dataset_name, token=args.catalog_token
        )
        return inspection, source_args
    if args.catalog_token is not None:
        raise ValueError("--catalog-token is only valid with --catalog-url or --dataset-url.")
    dataset_name = args.dataset_name or "rrd_dataset"
    source_args = ["--rrd-dir", str(args.rrd_dir)]
    return inspect_rrd_dataset(args.rrd_dir, dataset_name=dataset_name), source_args


def main() -> None:
    args = _parse_args()

    # Guided mode: without an explicit action/state/fps (or with --inspect), connect
    # to the dataset, show the convertible columns, and suggest a full command.
    if args.inspect or args.action is None or args.state is None or args.fps is None:
        inspection, source_args = _inspect_source(args)
        output = str(args.output) if args.output is not None else "/path/to/output/dataset"
        command = suggest_command(inspection, source_args=source_args, output=output)
        print(inspection.format_report(suggested_command=command))
        if not args.inspect:
            missing = [name for name, val in (("--action", args.action), ("--state", args.state)) if val is None]
            if args.fps is None:
                missing.append("--fps")
            print(f"\nMissing {', '.join(missing)}. Re-run with the flags above to convert.")
        return

    if args.output is None:
        raise ValueError("--output is required to convert.")

    video_specs = _parse_video_specs(args.video)
    config = LeRobotConversionConfig(
        fps=args.fps,
        index_column=args.index,
        action=args.action,
        state=args.state,
        task=args.task,
        videos=video_specs,
        use_videos=not args.use_images,
        action_names=_parse_name_list(args.action_names),
        state_names=_parse_name_list(args.state_names),
    )

    if args.dataset_url is not None:
        # repo_id defaults to the entry id (the last URL path segment).
        repo_id = args.repo_id or args.dataset_url.rstrip("/").split("/")[-1]
        convert_dataset_url_to_lerobot(
            dataset_url=args.dataset_url,
            token=args.catalog_token,
            output_dir=args.output,
            repo_id=repo_id,
            config=config,
            segments=args.segments,
            max_segments=args.max_segments,
        )
    elif args.catalog_url is not None:
        if not args.dataset_name:
            raise ValueError("--dataset-name is required when using --catalog-url.")
        repo_id = args.repo_id or args.dataset_name
        convert_catalog_dataset_to_lerobot(
            catalog_url=args.catalog_url,
            dataset_name=args.dataset_name,
            token=args.catalog_token,
            output_dir=args.output,
            repo_id=repo_id,
            config=config,
            segments=args.segments,
            max_segments=args.max_segments,
        )
    else:
        if args.catalog_token is not None:
            raise ValueError("--catalog-token is only valid with --catalog-url or --dataset-url.")
        dataset_name = args.dataset_name or "rrd_dataset"
        repo_id = args.repo_id or dataset_name
        convert_rrd_dataset_to_lerobot(
            rrd_dir=args.rrd_dir,
            output_dir=args.output,
            dataset_name=dataset_name,
            repo_id=repo_id,
            config=config,
            segments=args.segments,
            max_segments=args.max_segments,
        )


if __name__ == "__main__":
    main()
