# Copyright (c) 2026 Dexteleop Intelligence (Lingyu Intelligence)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""Write a batch of episodes produced by MCAPSampleExtractor as one Iceberg snapshot.

Save pipeline, two stages, across processes::
    worker process  EpisodeParquetWriter.write(): one episode -> one row -> one row group
                                               close the file once roll_episodes rows accumulate and hand over the path
    main process    IcebergEpisodeSaver.extend(): table.add_files(path) -> one snapshot

Why two stages: Arrow conversion is 95% of the save cost and pure Python holding the GIL -- measured
multithreading speedup was zero (1.00x); only worker processes truly parallelize. add_files only registers files
without rewriting data, so one main-process call takes just 0.12 s. Queues therefore only carry file paths,
Why episodes are written incrementally rather than batched: the worker's memory thus becomes independent of file
row count, staying constant at one episode (measured RSS steady at +0.11 GB); files can grow as large as desired and the commit count kept very low.

One row is one episode::
    episode_id          string  = "{source_id}:{source_episode_seq:08d}", globally stable and unique, for retry deduplication
    source_id           string  = mcap file name (without extension)
    source_episode_seq  int64   = the X/Y pair sequence number within that mcap
    num_samples         int64   = len(samples), stored redundantly so the final numbering stage need not read the huge samples column
    samples             LIST<STRUCT<topic: STRUCT<msg_type, msg_def, log_time, locations>>>

    num_samples         int64   = len(samples), stored redundantly so the final numbering stage need not read the huge samples column
on the final snapshot, ordered by (source_id, source_episode_seq).

Data is split by "robot/task": the catalog name is the current robot (ROBOT selected in mcap_config),
and the namespace name derives from the mcap's prompt (task), so one robot's tasks each get a namespace.
Not merging into one big table: samples' STRUCT schema depends on the robot's topic set, and mixing tables would force a union
schema; cross-task global numbering depends only on metadata columns and is done uniformly by build_sample_idx.py at index time.

Usage::
    table = load_episodes_table(warehouse_dir, topic_names, prompt)
    # in a worker process
    writer = EpisodeParquetWriter(table, topic_names, roll_episodes)
    closed = writer.write(EpisodeRecord(...))   # Returns (path, episode count) when a file fills up
    closed = writer.close()                     # Wrap up, hand over the last not-yet-full file
    # in the main process
    IcebergEpisodeSaver(table).extend([path], num_episodes)

How many episodes one parquet accumulates is the caller's decision; see build_sample_idx.EPISODES_PER_PARQUET.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.table import Table

from openpi.training.lingyu_dataloader_v2.mcap_config.config import ROBOT

# Constants
EPISODES_TABLE_NAME = "episodes"

# A message's locating primary key; field names map one to one to MCAP_Player's location quadruple
LOCATION_TYPE = pa.struct([
    pa.field("mcap_url", pa.string(), nullable=False),
    pa.field("chunk_file_offset", pa.int64(), nullable=False),
    pa.field("uncompressed_byte_offset", pa.int64(), nullable=False),
    pa.field("record_length", pa.int64(), nullable=False),
])

# The messages a topic selects on one sample; the location count varies by role (obs one GOP, action one chunk)
TOPIC_MESSAGE_TYPE = pa.struct([
    pa.field("msg_type", pa.string(), nullable=False),
    pa.field("msg_def", pa.string(), nullable=False),
    pa.field("log_time", pa.int64(), nullable=False),
    pa.field("locations", pa.list_(LOCATION_TYPE), nullable=False),
])


def build_episodes_schema(topic_names: Sequence[str]) -> pa.Schema:
    """Build the Iceberg/Arrow schema of one episode row for the given topics."""
    # A sample is a STRUCT with a fixed topic set; a missing topic is written as None, hence nullable
    sample_type = pa.struct([
        pa.field(topic_name, TOPIC_MESSAGE_TYPE, nullable=True)
        for topic_name in topic_names
    ])
    return pa.schema([
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_episode_seq", pa.int64(), nullable=False),
        pa.field("num_samples", pa.int64(), nullable=False),
        pa.field("samples", pa.list_(sample_type), nullable=False),
    ])


def prompt_to_namespace(prompt: str) -> str:
    """Turn a prompt into a valid Iceberg namespace, e.g. "Fold the towels" -> "fold_the_towels"."""
    return re.sub(r"[^0-9a-z]+", "_", prompt.lower()).strip("_")


def load_episodes_catalog(warehouse_dir: str) -> Catalog:
    """Open (creating the warehouse dir on first use) the current robot's sqlite-backed catalog."""
    # The catalog is this function's responsibility, not the saver's; the saver only takes a Table, decoupled from sqlite/REST/S3 environments
    warehouse_path = Path(warehouse_dir)
    warehouse_path.mkdir(parents=True, exist_ok=True)
    # One catalog.db distinguishes robots by catalog name, so switching robots needs no new warehouse directory
    return load_catalog(
        ROBOT,
        **{"type": "sql",
           "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
           "warehouse": f"file://{warehouse_path}"},
    )


def load_episodes_table(warehouse_dir: str, topic_names: Sequence[str], prompt: str) -> Table:
    """Load (creating on first use) the episodes table of one prompt (task) of the current robot."""
    catalog = load_episodes_catalog(warehouse_dir)
    namespace = prompt_to_namespace(prompt)
    # Namespace names are irreversible after normalization, so the original prompt text is stored in namespace attributes and recoverable at training time
    catalog.create_namespace_if_not_exists(namespace, properties={"prompt": prompt})
    return catalog.create_table_if_not_exists(
        f"{namespace}.{EPISODES_TABLE_NAME}", schema=build_episodes_schema(topic_names))


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode as produced by MCAPSampleExtractor.iter_episodes()."""
    source_id: str
    source_episode_seq: int
    # [{topic: [msg_type, msg_def, log_time, [location, ...]]}, ...], the index being sample_idx
    samples: list[dict[str, Any]]

    @property
    def episode_id(self) -> str:
        """Stable unique identity, so a retried commit can be de-duplicated later."""
        return f"{self.source_id}:{self.source_episode_seq:08d}"


def _message_to_arrow(message: Sequence[Any]) -> dict[str, Any]:
    """Turn one [msg_type, msg_def, log_time, locations] record into named Arrow fields."""
    msg_type, msg_def, log_time, locations = message
    return {
        "msg_type": msg_type,
        "msg_def": msg_def,
        "log_time": int(log_time),   # log_time is a nanosecond string; stored as int64 without precision loss
        "locations": [
            # Name the quadruple's positions so SQL can write location.mcap_url directly afterwards
            dict(zip(LOCATION_TYPE.names, (url, int(chunk), int(offset), int(length))))
            for url, chunk, offset, length in locations
        ],
    }


def _episode_to_row(episode: EpisodeRecord, topic_names: Sequence[str]) -> dict[str, Any]:
    """Turn one EpisodeRecord into one Arrow row matching build_episodes_schema()."""
    return {
        "episode_id": episode.episode_id,
        "source_id": episode.source_id,
        "source_episode_seq": int(episode.source_episode_seq),
        "num_samples": len(episode.samples),
        # Every sample is padded to the same topic key set, missing ones written as None, keeping every row's STRUCT schema identical
        "samples": [
            {topic_name: (_message_to_arrow(sample[topic_name])
                          if topic_name in sample else None)
             for topic_name in topic_names}
            for sample in episode.samples
        ],
    }


class EpisodeParquetWriter:
    """Append episodes to one parquet file, rolling to a new file every roll_episodes.

    Meant to run inside the worker process: from_pylist is pure Python and holds the
    GIL, so it only goes parallel across processes, never across threads.
    One episode is one row, written as its own row group, so the worker only ever holds
    a single episode in memory no matter how many rows the file ends up with.
    A file is not part of the table until the parent registers it via add_files().
    """

    def __init__(self, table: Table, topic_names: Sequence[str], roll_episodes: int):
        assert roll_episodes > 0, f"roll_episodes 必须为正: {roll_episodes}"
        self._table = table
        self._topic_names = tuple(topic_names)
        self._roll_episodes = roll_episodes
        self._arrow_schema = build_episodes_schema(self._topic_names)

        self._writer: pq.ParquetWriter | None = None
        self._output_stream = None
        self._path = ""
        self._num_episodes = 0      # Episodes written into the current file

    def write(self, episode: EpisodeRecord) -> tuple[str, int] | None:
        """Write one episode; return (path, episode count) when the file just rolled."""
        if self._writer is None:
            # uuid file name: dozens of workers write the same data directory concurrently; it guarantees no overwriting
            self._path = f"{self._table.location().rstrip('/')}/data/{uuid4()}.parquet"
            self._output_stream = self._table.io.new_output(self._path).create(overwrite=False)
            self._writer = pq.ParquetWriter(self._output_stream, self._arrow_schema,
                                            compression="zstd")
        # One row per row group: training fetches by episode, so reading one row group suffices, with no read amplification
        self._writer.write_table(pa.Table.from_pylist(
            [_episode_to_row(episode, self._topic_names)], schema=self._arrow_schema))
        self._num_episodes += 1
        return self.close() if self._num_episodes >= self._roll_episodes else None

    def close(self) -> tuple[str, int] | None:
        """Finish the current file (if any) and return (path, episode count)."""
        if self._writer is None:
            return None
        self._writer.close()
        self._output_stream.close()
        self._writer, self._output_stream = None, None
        # Take it then reset to zero: the next episode opens a new file
        closed = (self._path, self._num_episodes)
        self._num_episodes = 0
        return closed


class IcebergEpisodeSaver:
    """Register worker-written parquet files into the table, one snapshot per call."""

    def __init__(self, table: Table):
        self._table = table
        self.num_episodes = 0       # Total episodes registered
        self.num_commits = 0        # Total snapshots produced

    def extend(self, parquet_paths: Sequence[str], num_episodes: int) -> None:
        """Register already-written parquet files as one Iceberg snapshot."""
        # add_files only registers files without rewriting data; uuid paths cannot repeat, so that full-table scan is disabled
        self._table.add_files(
            file_paths=list(parquet_paths),
            snapshot_properties={"writer": "lingyu-episode-saver",
                                 "episode-count": str(num_episodes)},
            check_duplicate_files=False,
        )
        self.num_episodes += num_episodes
        self.num_commits += 1
