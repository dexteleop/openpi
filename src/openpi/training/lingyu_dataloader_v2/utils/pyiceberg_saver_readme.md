对。保存阶段可以完全不用 DuckDB。

你现在可以把职责定成：

```text
保存阶段：
Python objects
   ↓
PyArrow Table（16 行）
   ↓
PyIceberg table.append()
   ↓
Parquet data files
   ↓
Iceberg snapshot commit

最终编号阶段：
Iceberg
   ↓
DuckDB
   ↓
ORDER BY + ROW_NUMBER()
   ↓
episode_offset
```

PyIceberg 当前官方 API 直接支持 `table.append(pa.Table)`，一次 append 会把这个 Arrow batch 写入数据文件并产生一次 Iceberg snapshot commit。([PyIceberg][1])

你的情况：

```text
16 episodes
=
16 Iceberg rows
=
1 个 PyArrow Table
=
1 次 table.append()
=
1 次 Iceberg commit
```

物理上可能产生一个或多个 Parquet 文件，但这一点不要让业务代码关心。

## 1. 我建议的最终 schema

你原来的 episode 数据是：

```text
episode
=
LIST<
    sample
>
```

sample 是固定 topic 的 STRUCT：

```text
sample
=
STRUCT<
    topic_a: topic_message,
    topic_b: topic_message,
    topic_c: topic_message,
    ...
>
```

每个 message：

```text
topic_message
=
STRUCT<
    msg_type,
    msg_def,
    log_time,
    locations
>
```

locations：

```text
LIST<
    STRUCT<
        mcap_url,
        chunk_offset,
        uncompressed_bytes_offset,
        record_length
    >
>
```

我建议 Iceberg 顶层一行不要**只有** episode，而是：

```text
Iceberg row
│
├── episode_id
├── source_id
├── source_episode_seq
│
└── episode
     │
     └── LIST<sample>
```

也就是：

```text
episode_id           string
source_id            string
source_episode_seq   int64

episode              LIST<
    STRUCT<
        topic_a: STRUCT<...>,
        topic_b: STRUCT<...>,
        ...
    >
>
```

原因是未来生成：

```text
episode_offset = 1...N
```

的时候，你必须拥有稳定的 episode 身份以及排序依据。

例如最终可以：

```sql
ROW_NUMBER() OVER (
    ORDER BY source_id, source_episode_seq
)
```

千万不要使用 Parquet row position 或 Iceberg commit 顺序作为 episode ID。

---

## 2. `locations` 不建议真的存 Tuple

你的 Python 数据可以继续是：

```python
(
    mcap_url,
    chunk_offset,
    uncompressed_bytes_offset,
    record_length,
)
```

但是写 Iceberg 时，我建议转换成：

```python
{
    "mcap_url": ...,
    "chunk_offset": ...,
    "uncompressed_bytes_offset": ...,
    "record_length": ...,
}
```

也就是 Iceberg：

```text
LIST<STRUCT<...>>
```

而不是把 tuple 本身当 schema。

这样字段有名字，以后查询非常舒服：

```sql
location.mcap_url
location.chunk_offset
```

---

# 3. 一个完整 Python 模板

先确保有 PyArrow。PyIceberg 官方文档说明 Arrow 读写需要 `pyarrow`；安装 PyIceberg 时也可以通过对应 extra 安装。([PyIceberg][2])

```bash
pip install pyarrow
```

然后可以这样写。

```python
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa

from pyiceberg.catalog import Catalog
from pyiceberg.table import Table


# ============================================================
# 1. Schema
# ============================================================

LOCATION_TYPE = pa.struct([
    pa.field("mcap_url", pa.string(), nullable=False),
    pa.field("chunk_offset", pa.int64(), nullable=False),
    pa.field("uncompressed_bytes_offset", pa.int64(), nullable=False),
    pa.field("record_length", pa.int64(), nullable=False),
])


TOPIC_MESSAGE_TYPE = pa.struct([
    pa.field("msg_type", pa.string(), nullable=False),
    pa.field("msg_def", pa.string(), nullable=False),

    # 假设你的 log_time 是整数纳秒时间戳。
    # 用 int64 可以原样保存，不损失精度。
    pa.field("log_time", pa.int64(), nullable=False),

    pa.field(
        "locations",
        pa.list_(LOCATION_TYPE),
        nullable=False,
    ),
])


def build_sample_type(topic_names: Sequence[str]) -> pa.StructType:
    """
    一个 sample 是固定 topic 集合组成的 STRUCT。

    topic 可以在某一个 sample 中不存在，所以各 topic field nullable=True。
    """
    return pa.struct([
        pa.field(
            topic_name,
            TOPIC_MESSAGE_TYPE,
            nullable=True,
        )
        for topic_name in topic_names
    ])


def build_episode_table_schema(
    topic_names: Sequence[str],
) -> pa.Schema:
    sample_type = build_sample_type(topic_names)

    return pa.schema([
        # 顶层 metadata。
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),

        # source 内 episode 的稳定序号。
        pa.field("source_episode_seq", pa.int64(), nullable=False),

        # 真正的 episode payload。
        pa.field(
            "episode",
            pa.list_(sample_type),
            nullable=False,
        ),
    ])
```

如果 topic 是：

```python
TOPIC_NAMES = [
    "/camera/front",
    "/joint_states",
    "/cmd_vel",
]
```

最终逻辑结构就是：

```text
row
├── episode_id
├── source_id
├── source_episode_seq
│
└── episode: LIST
     │
     ├── sample 0: STRUCT
     │    ├── /camera/front
     │    ├── /joint_states
     │    └── /cmd_vel
     │
     ├── sample 1
     └── ...
```

---

# 4. 定义输入数据

你的程序可以继续使用普通 Python 对象。

```python
@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    source_id: str
    source_episode_seq: int

    # [
    #     {
    #         topic_name: {
    #             "msg_type": ...,
    #             "msg_def": ...,
    #             "log_time": ...,
    #             "locations": [
    #                 (url, chunk_offset, uncompressed_offset, record_length),
    #                 ...
    #             ]
    #         },
    #         ...
    #     },
    #     ...
    # ]
    samples: list[dict[str, Any]]
```

然后写一个 normalize，把你的 tuple locations 转成 Iceberg/Arrow 的 STRUCT：

```python
def normalize_location(location: tuple) -> dict[str, Any]:
    (
        mcap_url,
        chunk_offset,
        uncompressed_bytes_offset,
        record_length,
    ) = location

    return {
        "mcap_url": mcap_url,
        "chunk_offset": int(chunk_offset),
        "uncompressed_bytes_offset": int(
            uncompressed_bytes_offset
        ),
        "record_length": int(record_length),
    }


def normalize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "msg_type": message["msg_type"],
        "msg_def": message["msg_def"],
        "log_time": int(message["log_time"]),
        "locations": [
            normalize_location(location)
            for location in message["locations"]
        ],
    }
```

然后 normalize sample。

因为 topic schema 是固定的，我建议确保**每一个 sample 都输出完全相同的 STRUCT schema**。

不存在的 topic 用 `None`：

```python
def normalize_sample(
    sample: Mapping[str, Any],
    topic_names: Sequence[str],
) -> dict[str, Any]:

    result = {}

    for topic_name in topic_names:
        message = sample.get(topic_name)

        if message is None:
            result[topic_name] = None
        else:
            result[topic_name] = normalize_message(message)

    return result
```

最后 episode：

```python
def episode_to_row(
    record: EpisodeRecord,
    topic_names: Sequence[str],
) -> dict[str, Any]:

    return {
        "episode_id": record.episode_id,
        "source_id": record.source_id,
        "source_episode_seq": record.source_episode_seq,

        "episode": [
            normalize_sample(sample, topic_names)
            for sample in record.samples
        ],
    }
```

---

# 5. 核心 EpisodeSaver

这里就是你真正需要的部分。

```python
class IcebergEpisodeSaver:
    """
    将 episode 缓存在内存中。

    batch_size 个 episode:
        -> 一个 PyArrow Table
        -> table.append()
        -> 一个 Iceberg commit

    默认 batch_size = 16。
    """

    def __init__(
        self,
        table: Table,
        topic_names: Sequence[str],
        batch_size: int = 16,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self._table = table
        self._topic_names = tuple(topic_names)
        self._batch_size = batch_size

        self._arrow_schema = build_episode_table_schema(
            self._topic_names
        )

        self._buffer: list[EpisodeRecord] = []
        self._lock = Lock()

    def append(self, episode: EpisodeRecord) -> None:
        """
        加入一个 episode。

        达到 batch_size 后自动 flush。
        """
        batch = None

        with self._lock:
            self._buffer.append(episode)

            if len(self._buffer) >= self._batch_size:
                batch = self._buffer[:self._batch_size]
                del self._buffer[:self._batch_size]

        if batch is not None:
            self._commit_batch(batch)

    def flush(self) -> None:
        """
        把不足 batch_size 的剩余 episode 也写出去。
        """
        with self._lock:
            if not self._buffer:
                return

            batch = self._buffer
            self._buffer = []

        self._commit_batch(batch)

    def close(self) -> None:
        self.flush()

    def _commit_batch(
        self,
        episodes: Sequence[EpisodeRecord],
    ) -> None:

        rows = [
            episode_to_row(
                record=episode,
                topic_names=self._topic_names,
            )
            for episode in episodes
        ]

        # 这里产生的 Table：
        #
        # row 0  = episode 0
        # row 1  = episode 1
        # ...
        # row 15 = episode 15
        #
        arrow_table = pa.Table.from_pylist(
            rows,
            schema=self._arrow_schema,
        )

        assert arrow_table.num_rows == len(episodes)

        # ==============================================
        # 真正的 Iceberg append + commit
        # ==============================================
        self._table.append(
            arrow_table,
            snapshot_properties={
                "writer": "robot-episode-saver",
                "episode-count": str(len(episodes)),
            },
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
```

PyIceberg 官方 API 的 `Table.append()` 接受一个 materialized `pa.Table`，并允许给 snapshot 附加自定义 properties，所以这里可以把 `episode-count=16` 写进 snapshot summary。([PyIceberg][1])

---

# 6. 创建 Iceberg table

我建议 saver 本身不要负责创建 catalog。

也就是说：

```text
Catalog configuration
        ↓
load table
        ↓
EpisodeSaver(table)
```

这样 saver 和 REST/S3/MinIO 等环境完全解耦。

比如生产环境使用 REST Catalog：

```python
from pyiceberg.catalog import load_catalog


catalog = load_catalog(
    "robot",
    type="rest",
    uri="http://localhost:8181",
    warehouse="robot_warehouse",
)
```

PyIceberg 当前原生提供 REST Catalog，并通过 Catalog 进行 table load/commit。([PyIceberg][3])

定义：

```python
TOPIC_NAMES = [
    "/camera/front",
    "/joint_states",
    "/cmd_vel",
]

ARROW_SCHEMA = build_episode_table_schema(
    TOPIC_NAMES
)
```

初始化 namespace：

```python
catalog.create_namespace_if_not_exists(
    "robot_data"
)
```

创建 table：

```python
table = catalog.create_table_if_not_exists(
    "robot_data.episodes",
    schema=ARROW_SCHEMA,
)
```

PyIceberg 的 catalog API 可以直接接受 PyArrow schema 来创建 Iceberg table，所以这里不需要你手工写 Iceberg 的 `NestedField(field_id=...)`。([PyIceberg][2])

---

# 7. 实际保存 episode

假设一个 episode：

```python
episode = EpisodeRecord(
    episode_id="robot-003-episode-000001",
    source_id="robot-003",
    source_episode_seq=1,

    samples=[
        # sample 0
        {
            "/camera/front": {
                "msg_type": "sensor_msgs/Image",
                "msg_def": "...",
                "log_time": 1723456789000000000,
                "locations": [
                    (
                        "s3://robot-data/a.mcap",
                        1024,
                        532,
                        4096,
                    )
                ],
            },

            "/joint_states": {
                "msg_type": "sensor_msgs/JointState",
                "msg_def": "...",
                "log_time": 1723456789000000100,
                "locations": [
                    (
                        "s3://robot-data/a.mcap",
                        1024,
                        4628,
                        500,
                    )
                ],
            },
        },

        # sample 1
        {
            "/camera/front": {
                "msg_type": "sensor_msgs/Image",
                "msg_def": "...",
                "log_time": 1723456789033333333,
                "locations": [
                    (
                        "s3://robot-data/a.mcap",
                        8192,
                        100,
                        4096,
                    )
                ],
            },
        },
    ],
)
```

然后：

```python
saver = IcebergEpisodeSaver(
    table=table,
    topic_names=TOPIC_NAMES,
    batch_size=16,
)

saver.append(episode)
```

不断：

```python
for episode in generated_episodes:
    saver.append(episode)

saver.close()
```

假设一共收到：

```text
150 episodes
```

那么：

```text
16
16
16
16
16
16
16
16
16
6
```

即大约：

```text
10 次 append / commit
```

而不是：

```text
150 次 commit
```

---

# 8. 你的 1000 个 source 怎么接这个 saver

我不会让 1000 个 source 都拥有自己的 `IcebergEpisodeSaver`。

更建议：

```text
Source 1 ─────┐
Source 2 ─────┤
Source 3 ─────┤
...           │
Source 1000 ──┘
              │
              ▼
       Episode Queue
              │
              ▼
      IcebergEpisodeSaver
              │
         每 16 个
              │
              ▼
      table.append()
```

也就是说 source 只负责：

```python
episode_queue.put(episode)
```

而一个或少数几个 ingestion workers 做：

```python
episode = episode_queue.get()
saver.append(episode)
```

这样会比 1000 个 source 自己争抢 Iceberg commit 简单很多。

---

## 9. 有一个并发安全问题要注意

上面的基础模板用了：

```python
Lock
```

保证内存 buffer 不会被多个 producer 同时破坏。

但是对你真正的生产系统，我更推荐：

```text
producer threads/processes
        ↓
thread/process safe Queue
        ↓
专门 Iceberg writer
```

而不是所有 producer 直接调用：

```python
saver.append()
```

因为真正耗时的是：

```python
table.append()
```

里面包含：

```text
Arrow
 ↓
Parquet write
 ↓
manifest
 ↓
Iceberg catalog commit
```

不应该阻塞数据源本身。

---

# 10. 一个需要特别处理的失败场景

当前 PyIceberg 明确区分了：

```python
CommitFailedException
```

和：

```python
CommitStateUnknownException
```

前者表示 commit 没成功，可以 refresh/retry；后者意味着客户端无法确认 commit 到底成功还是失败。([PyIceberg][4])

因此生产代码不能简单：

```python
try:
    table.append(batch)
except:
    table.append(batch)
```

因为网络在：

```text
server commit 成功
        ↓
响应返回途中断线
```

时，客户端可能认为失败，如果无脑重试，就可能把相同的 16 个 episode 再 append 一遍。

所以 **`episode_id` 必须全局稳定且唯一**：

```text
episode_id =
source_id + source_episode_seq
```

例如：

```text
robot-00023:00000142
```

这不是为了排序，主要是为了：

```text
重试
故障恢复
去重
数据审计
```

Iceberg 不会自动替你强制 `episode_id UNIQUE`。

---

# 11. 一个我会做的小修改：不要叫 `episode`，叫 `samples`

从数据模型角度：

```text
一行本身已经是 Episode
```

所以这一列：

```python
episode: LIST<sample>
```

虽然没错，但会出现语义：

```text
Episode row
    └── episode
```

我个人更推荐：

```text
Episode row
│
├── episode_id
├── source_id
├── source_episode_seq
│
└── samples
```

也就是：

```python
pa.field(
    "samples",
    pa.list_(sample_type),
    nullable=False,
)
```

最终 schema 会更自然：

```text
episodes table

episode_id
source_id
source_episode_seq
samples
```

对应：

```text
row = Episode
samples = LIST<Sample>
sample = STRUCT<Topics>
topic = STRUCT<Message>
locations = LIST<Location>
```

整个类型树就是：

```text
Iceberg Table
│
└── Row = Episode
     │
     ├── episode_id: STRING
     ├── source_id: STRING
     ├── source_episode_seq: LONG
     │
     └── samples: LIST
          │
          └── Sample: STRUCT
               │
               ├── topic_A: STRUCT
               │    ├── msg_type
               │    ├── msg_def
               │    ├── log_time
               │    └── locations: LIST
               │         └── STRUCT
               │              ├── mcap_url
               │              ├── chunk_offset
               │              ├── uncompressed_bytes_offset
               │              └── record_length
               │
               ├── topic_B
               └── ...
```

这就是我建议你第一版实际实现的模型。

另外，**我暂时不会把 `episode_offset` 存进这张 raw Iceberg 表**。它属于 finalize 阶段，由 DuckDB 在最终 snapshot 上产生；保存器只负责可靠保存 `episode_id/source_id/source_episode_seq/samples`。这样采集路径会简单很多。

[1]: https://py.iceberg.apache.org/reference/pyiceberg/table/?utm_source=chatgpt.com "table - PyIceberg"
[2]: https://py.iceberg.apache.org/api/?utm_source=chatgpt.com "API - PyIceberg"
[3]: https://py.iceberg.apache.org/reference/pyiceberg/catalog/rest/?utm_source=chatgpt.com "rest - PyIceberg"
[4]: https://py.iceberg.apache.org/reference/pyiceberg/exceptions/?utm_source=chatgpt.com "exceptions - PyIceberg"
