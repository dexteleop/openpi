"""
用 MCAP_Player 播放 mcap, 三个视频 topic 各取 num_P = 0/5/10/15 的 message
(num_P = len(locations) - 1, 即当前帧之前需要先解的 P 帧数),
用 MCAP_Message_Fetcher 按 locations 取回 packets, 交给 cpu_decode_current_frame() 解码出当前帧,
并把图像保存到本地 png。
"""
import os

from PIL import Image

from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import MCAP_Message_Fetcher
from openpi.training.lingyu_dataloader_v2.utils.mcap_player import MCAP_Player
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import find_mcap_url_and_prompt_pairs
from openpi.training.lingyu_dataloader_v2.utils.video_frame_decoder import cpu_decode_current_frame
from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_mcap_video_topics_gop
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_TESTED_P_COUNTS = (0, 5, 10, 15)  # 待测的 P 帧数: 0 为纯 IDR, 其余需回溯解码
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "video_frame_decoder_output")


def test():
    """三个视频 topic 在 num_P = 0/5/10/15 下都能解出当前帧, 并保存成 png。"""
    path = next(iter(find_mcap_url_and_prompt_pairs()))
    video_topics = set(load_mcap_video_topics_gop())
    fetcher = MCAP_Message_Fetcher()
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    logger.info("播放 %s, 视频 topic: %s, 待测 num_P: %s",
                path, sorted(video_topics), _TESTED_P_COUNTS)

    decoded_samples_idx = set()  # 已解码的 (topic_name, num_P)
    wanted_total = len(video_topics) * len(_TESTED_P_COUNTS)
    for message in MCAP_Player(path).play_messages():
        topic_name, msg_type, msg_def, log_time, locations = message
        num_p = len(locations) - 1  # 首个 location 是 IDR, 其余都是 P 帧

        # 只取视频 topic 中尚未测过的 num_P
        if (topic_name not in video_topics
            or num_p not in _TESTED_P_COUNTS
            or (topic_name, num_p) in decoded_samples_idx):
            continue

        # 测试是否能够正常解出图片
        frame = cpu_decode_current_frame(fetcher.fetch_message(msg_type, msg_def, locations))
        assert frame.ndim == 3 and frame.shape[2] == 3, f"{topic_name} 解码结果异常: {frame.shape}"

        png_path = os.path.join(
            _OUTPUT_DIR,
            f"{topic_name.strip('/').replace('/', '_')}_P{num_p:02d}_{log_time}.png")
        Image.fromarray(frame).save(png_path)
        logger.info("%s: num_P=%d -> %s, shape=%s, dtype=%s, 像素均值=%.1f",
                    topic_name, num_p, png_path, frame.shape, frame.dtype, frame.mean())
        assert os.path.getsize(png_path) > 0

        decoded_samples_idx.add((topic_name, num_p))
        if len(decoded_samples_idx) == wanted_total:
            break
    assert len(decoded_samples_idx) == wanted_total, f"未解码全部样本: {sorted(decoded_samples_idx)}"


if __name__ == "__main__":
    test()
