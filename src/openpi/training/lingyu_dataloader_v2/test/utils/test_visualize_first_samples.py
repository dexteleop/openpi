"""可视化 iceberg_warehouse 索引中第一个 episode 的前 5 个 sample.

数据结构(全部在本地 iceberg_warehouse 内, 不依赖 /mnt/data 挂载, 不回读 mcap 原数据):

  episode_index.parquet   每行一个 episode 的定位信息
      prompt               提示词
      episode_id           episode 唯一标识 (source_id:八位序号)
      source_id            来源 rosbag 录制批次
      source_episode_seq   该批次内的 episode 序号
      num_samples          该 episode 的 sample 总数
      data_file            数据 parquet 的 file:// 路径
      row_group            episode 在数据 parquet 中的 row group 号
      sample_offset        该 episode 首个 sample 的全局编号

  episodes/data/*.parquet 一行(一个 row group)一个 episode, samples 列是 list, 每 sample:
      {topic: {msg_type, msg_def, log_time, locations}}
      msg_type / msg_def   ROS2 消息类型与定义
      log_time             该 sample 该 topic 的采样时刻(纳秒)
      locations            该 sample 该 topic 需要读取的全部 mcap 记录定位, 每条四元组:
          mcap_url                  mcap 文件路径(本地绝对路径或 OSS key)
          chunk_file_offset         记录所在 chunk 在 mcap 文件内的起始偏移
          uncompressed_byte_offset  记录在 chunk 解压后数据中的起始偏移
          record_length             记录内容字节数

  每 sample 固定 21 个 topic, 分三类(location 数规律):
      视频   3 个: 覆盖当前帧的完整 GOP(GOP=45), location 数随 I 帧位置在 [1, 45] 变化
      state  6 个: 当前时刻一条消息, 恒 1 个 location
      action 12 个: 未来 ACTION_CHUNK_LENGTH=100 步动作序列, 恒 100 个 location

本测试做三件事:
  1. 上述索引明细逐 sample 写入本文件同名 .log(含一条 location 四元组示例),
     每 topic 一行, 各列含义:
      topic        ROS2 话题名(每 sample 固定 21 个)
      msg_type     该 topic 消息的 ROS2 类型, 解码时按它选反序列化器
      loc          location 条数 = 需从 mcap 回读的记录数
                   (state 恒 1 / action 恒 100 / 视频 1~45 随 I 帧位置变)
      bytes        全部 location 的 record_length 之和, 即回读的内容字节总量
      t(s)         该 topic 的 log_time 相对 sample 0 视频帧时刻的秒数(纳秒换算)
      mcap_files   这些 location 涉及的 mcap 文件名(mcap_url 取 basename 去重)
  2. 21 topic(行) x 5 sample(列) 的 locations 数表格存
     visualize_output/first_samples_index_table.png
  3. 断言三类 topic 的 location 数不变式, 作为索引结构的回归检查

运行: cd ~/openpi && uv run pytest src/openpi/training/lingyu_dataloader_v2/test/utils/test_visualize_first_samples.py -s
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from openpi.training.lingyu_dataloader_v2.build_sample_idx import WAREHOUSE_DIR
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import (
    load_episode_index,
    load_sample_topics)
from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_mcap_video_topics_gop
from openpi.training.lingyu_dataloader_v2.model_config.config import load_state_and_action_concat
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

# 可视化的 sample 数量
VISUALIZE_SAMPLE_COUNT = 5
OUTPUT_DIR = Path(__file__).parent / "visualize_output"
# location 四元组示例取这个 topic 的首条(state 类, 恒 1 条, 展示最直观)
EXAMPLE_TOPIC = "/left_arm/joint_states"


def log_column_meanings() -> None:
    """把明细各列含义写入日志开头, 让 .log 脱离代码也能看懂(内容与文件头 docstring 一致)."""
    logger.info("明细各列含义:")
    logger.info("  topic        ROS2 话题名(每 sample 固定 21 个)")
    logger.info("  msg_type     该 topic 消息的 ROS2 类型, 解码时按它选反序列化器")
    logger.info("  loc          location 条数 = 需从 mcap 回读的记录数")
    logger.info("               (state 恒 1 / action 恒 100 / 视频 1~45 随 I 帧位置变)")
    logger.info("  bytes        全部 location 的 record_length 之和, 即回读的内容字节总量")
    logger.info("  t(s)         该 topic 的 log_time 相对 sample 0 视频帧时刻的秒数(纳秒换算)")
    logger.info("  mcap_files   这些 location 涉及的 mcap 文件名(mcap_url 取 basename 去重)")


def log_location_example(sample_topics: dict) -> None:
    """打印一条完整的 location 四元组, 展示索引中 location 的字段含义."""
    example_location = sample_topics[EXAMPLE_TOPIC]["locations"][0]
    logger.info("location 示例 (%s 的唯一一条):", EXAMPLE_TOPIC)
    logger.info("  mcap_url                 = %s", example_location["mcap_url"])
    logger.info("  chunk_file_offset        = %d (记录所在 chunk 的文件内偏移)",
                example_location["chunk_file_offset"])
    logger.info("  uncompressed_byte_offset = %d (记录在 chunk 解压数据中的偏移)",
                example_location["uncompressed_byte_offset"])
    logger.info("  record_length            = %d (记录内容字节数)",
                example_location["record_length"])


def log_sample_index(sample_idx: int, sample_topics: dict, base_log_time: int) -> None:
    """把一个 sample 的索引元数据逐 topic 写入日志; 时刻为相对 sample 0 的秒数."""
    logger.info("=== sample_idx=%d: %d 个 topic ===", sample_idx, len(sample_topics))
    # 列头与下方数据行同宽对齐: 主题名 / 消息类型 / location数 / 总字节 / 相对时刻(s) / 涉及的mcap文件
    logger.info("  %-30s %-45s %3s %8s %5s  %s",
                "topic", "msg_type", "loc", "bytes", "t(s)", "mcap_files")
    for topic, message in sample_topics.items():
        topic_locations = message["locations"]
        bytes_total = sum(location["record_length"] for location in topic_locations)
        mcap_files = {Path(location["mcap_url"]).name for location in topic_locations}
        relative_time = (message["log_time"] - base_log_time) / 1e9
        logger.info("  %-30s %-45s %3d locs %8d B %5.2fs  %s",
                    topic, message["msg_type"], len(topic_locations),
                    bytes_total, relative_time, ", ".join(sorted(mcap_files)))


def assert_topic_invariants(sample_topics: dict, video_gop: dict,
                            state_fields: dict, action_fields: dict) -> None:
    """断言三类 topic 的 location 数不变式: state 恒 1, action 恒 100, 视频在 [1, GOP]."""
    for topic in state_fields:
        assert len(sample_topics[topic]["locations"]) == 1, \
            f"state topic {topic} 应恒 1 个 location"
    for topic in action_fields:
        assert len(sample_topics[topic]["locations"]) == 100, \
            f"action topic {topic} 应恒 100 个 location (ACTION_CHUNK_LENGTH)"
    for topic, gop_length in video_gop.items():
        locations_count = len(sample_topics[topic]["locations"])
        assert 1 <= locations_count <= gop_length, \
            f"视频 topic {topic} 的 location 数 {locations_count} 超出 [1, {gop_length}]"
    # 索引只存定位: 不应有任何字段携带消息原始内容
    for message in sample_topics.values():
        assert set(message) == {"msg_type", "msg_def", "log_time", "locations"}, \
            f"sample 字段集合异常: {set(message)}"


def save_index_table(samples_topics: list[dict], prompt: str, episode_id: str) -> Path:
    """画 topic(行) x sample(列) 的 locations 数表格, 存 PNG 并返回路径."""
    # 固定行序: 按 sample 0 的 topic 顺序, 其余 sample 的行对齐
    topic_order = list(samples_topics[0])
    cell_height, cell_width = 26, 150
    label_width = 330
    header_rows = 2   # 标题行 + 列头行
    table_width = label_width + cell_width * len(samples_topics)
    table_height = cell_height * (header_rows + len(topic_order))
    table_image = Image.new("RGB", (table_width, table_height), "white")
    draw = ImageDraw.Draw(table_image)
    try:
        font_cells = ImageFont.load_default(14)
        font_title = ImageFont.load_default(17)
    except TypeError:  # 旧版 PIL 的 load_default 不接受字号
        font_cells = font_title = ImageFont.load_default()

    def draw_cell(text: str, left: int, top: int, width: int, fill=None):
        """画一个带边框的单元格文字."""
        if fill:
            draw.rectangle((left, top, left + width, top + cell_height), fill=fill)
        draw.rectangle((left, top, left + width, top + cell_height), outline="#cccccc")
        draw.text((left + 6, top + 5), text, fill="black", font=font_cells)

    draw.text((8, 4), f"episode {episode_id}  prompt: {prompt}  (cell = locations count)",
              fill="black", font=font_title)
    draw_cell("topic", 0, cell_height, label_width)
    for sample_idx in range(len(samples_topics)):
        draw_cell(f"sample #{sample_idx}", label_width + cell_width * sample_idx,
                  cell_height, cell_width)

    # locations 数非 1 时加深底色, 一眼区分 state(1) / action(100) / 视频(GOP)
    for row_idx, topic in enumerate(topic_order):
        row_top = cell_height * (header_rows + row_idx)
        draw_cell(topic, 0, row_top, label_width)
        for sample_idx, sample_topics in enumerate(samples_topics):
            locations_count = len(sample_topics[topic]["locations"])
            shade = "#ffffff" if locations_count == 1 else "#dbe9ff"
            draw_cell(str(locations_count), label_width + cell_width * sample_idx,
                      row_top, cell_width, fill=shade)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "first_samples_index_table.png"
    table_image.save(output_path)
    return output_path


def test_visualize_first_samples():
    """展示第一个 episode 前 5 个 sample 的索引内容: 明细写日志, 表格存 PNG, 结构断言."""
    first_episode = load_episode_index(WAREHOUSE_DIR)[0]
    logger.info("episode=%s prompt=%s samples=%d data_file=%s",
                first_episode["episode_id"], first_episode["prompt"],
                first_episode["num_samples"], first_episode["data_file"])

    # 只读 parquet 行组, 全程不触碰 mcap_url 指向的原数据
    samples_topics = [load_sample_topics(first_episode["data_file"], first_episode["row_group"], sample_idx)
                      for sample_idx in range(min(VISUALIZE_SAMPLE_COUNT, first_episode["num_samples"]))]

    video_gop = load_mcap_video_topics_gop()
    state_concat, action_concat = load_state_and_action_concat()
    state_fields = {topic for topic, _ in state_concat}
    action_fields = {topic for topic, _ in action_concat}

    log_location_example(samples_topics[0])
    log_column_meanings()
    base_log_time = samples_topics[0]["/xr_video_topic/ffmpeg"]["log_time"]
    for sample_idx, sample_topics in enumerate(samples_topics):
        log_sample_index(sample_idx, sample_topics, base_log_time)
        assert_topic_invariants(sample_topics, video_gop, state_fields, action_fields)

    output_path = save_index_table(samples_topics, first_episode["prompt"],
                                   first_episode["episode_id"])
    logger.info("索引表格已保存: %s", output_path)
