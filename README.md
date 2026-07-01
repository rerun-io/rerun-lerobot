# rerun-lerobot

Convert [Rerun](https://rerun.io) RRD recordings into [LeRobot](https://github.com/huggingface/lerobot) v3 datasets.

`rerun-lerobot` uses the Rerun OSS server API to query and transform RRD files into the LeRobot v3
format used for imitation-learning training pipelines in PyTorch. It loads RRD files, infers data
types from the recordings, resamples all time series to a target frame rate, and writes a LeRobot v3
dataset. Video streams are efficiently remuxed without re-encoding.

This package started life as the [`rerun_export` example](https://github.com/rerun-io/rerun/tree/main/examples/python/rerun_export)
in the Rerun repository.

## Installation

```bash
pip install rerun-lerobot
```

### ⚠️ Dependency conflict with LeRobot

The conversion relies on the Rerun OSS server API (`rr.server.Server`), which requires
**`rerun-sdk >= 0.27`**. LeRobot currently pins **`rerun-sdk < 0.27`**, so a naive install will fail
to resolve. Override LeRobot's pin at install time.

With `uv`:

```toml
# pyproject.toml
[tool.uv]
override-dependencies = ["rerun-sdk>=0.27"]
```

Or on the command line:

```bash
uv pip install rerun-lerobot --override <(echo "rerun-sdk>=0.27")
```

With `pip`, install and then force the newer `rerun-sdk`:

```bash
pip install rerun-lerobot
pip install --upgrade "rerun-sdk>=0.27"
```

## Usage

The package installs a `rerun-lerobot` CLI that converts a directory of RRD recordings into a
LeRobot v3 dataset:

```bash
rerun-lerobot \
  --rrd-dir /path/to/recordings \
  --output /path/to/output/dataset \
  --dataset-name my_robot_dataset \
  --fps 10 \
  --index real_time \
  --action /action:Scalars:scalars \
  --state /observation/joint_positions:Scalars:scalars \
  --task /language_instruction:TextDocument:text \
  --video front:/camera/front
```

### Column specification format

Action, state, and task columns are specified as fully qualified columns:

```
entity_path:ComponentName:field_name
```

For example `/robot/action:Scalars:scalars`.

### Video specification format

Videos are specified as `key:path`:

- `key`: camera identifier (e.g. `front`, `wrist`)
- `path`: entity path to the video stream (e.g. `/camera/front`)

The converter expects a [`VideoStream`](https://www.rerun.io/docs/reference/types/archetypes/video_stream)
archetype at the specified paths.

### Full example

```bash
rerun-lerobot \
  --rrd-dir ./robot_recordings \
  --output ./lerobot_dataset \
  --dataset-name robot_demos \
  --fps 15 \
  --action /robot/action:Scalars:scalars \
  --state /robot/state:Scalars:scalars \
  --task /task:TextDocument:text \
  --video front:/camera/front \
  --video wrist:/camera/wrist \
  --action-names "joint_0,joint_1,joint_2,gripper" \
  --state-names "joint_0,joint_1,joint_2,gripper"
```

The output directory contains:

- `data/`: Parquet files with aligned time series data
- `videos/`: encoded video files (unless `--use-images` is passed)
- `meta/`: dataset metadata and episode information

## Python API

```python
from pathlib import Path

from rerun_lerobot import LeRobotConversionConfig, VideoSpec
from rerun_lerobot.__main__ import convert_rrd_dataset_to_lerobot

config = LeRobotConversionConfig(
    fps=15,
    index_column="real_time",
    action="/robot/action:Scalars:scalars",
    state="/robot/state:Scalars:scalars",
    task="/task:TextDocument:text",
    videos=[VideoSpec(key="front", path="/camera/front")],
)

convert_rrd_dataset_to_lerobot(
    rrd_dir=Path("./robot_recordings"),
    output_dir=Path("./lerobot_dataset"),
    dataset_name="robot_demos",
    repo_id="robot_demos",
    config=config,
)
```

## Development

```bash
uv sync
uv run pytest
uv run mypy rerun_lerobot
uv run ruff check
```

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.
