# 文件夹规范
## 代码基本运行逻辑
### 两种play_message的播放方式
#### 第一种：直接播放chunk

#### 第二种：读取元数据 Summary / ChunkIndex / MessageIndex
前提： MCAP 文件包含完整的 Summary / ChunkIndex / MessageIndex

| 你的字段                   | 来源                  | MCAP 字段               |
| -------------------------- | --------------------- | ----------------------- |
| `chunk_file_offset`        | `ChunkIndex`          | `chunk_start_offset`    |
| `uncompressed_byte_offset` | `MessageIndex`        | `records[i].offset`     |
| `records_length`            | Message record header | `uint64 content_length` |
| `topic_name`               | `Channel`             | `topic`                 |
| `msg_type`                 | `Schema`              | `name`                  |
| `msg_defs`                 | `Schema`              | `data`                  |
| `log_time`                 | `MessageIndex`        | `records[i].log_time`   |

```markdown
ChunkIndex
 ├── chunk_start_offset ───────────────> chunk_file_offset
 │
 └── message_index_offsets
          │
          ▼
     MessageIndex
       ├── channel_id ───────┐
       │                     │
       └── records[]         │
            ├── log_time ────┼────────> log_time
            └── offset ──────┼────────> uncompressed_byte_offset
                             │
                             ▼
                         Channel
                           ├── topic ─────────> topic_name
                           └── schema_id
                                  │
                                  ▼
                               Schema
                                 ├── name ─────> msg_type
                                 └── data ─────> msg_defs
```

## 功能测试
想要测试这个文件夹中的功能函数， 测试程序写入到
openpi/training/lingyu_dataloader_v2/test/utils中，
并按照这个文件夹中的格式来构建测试程序并使用pytest运行