"""
用 MCAP_Player 播放 mcap, 三个视频 topic 各取 num_P = 0/5/10/15 的 message
(num_P = len(locations) - 1, 即当前帧之前需要先解的 P 帧数),
用 MCAP_Message_Fetcher 按 locations 取回 packets, 交给 cpu_decode_current_frame() 解码出当前帧,
并把图像保存到本地 png。

test_gpu: 同样的 packets 经 GPU 解码进程解码、裁单目(含旋转)、缩放, 与 CPU 解码 + openpi 原有的裁剪与缩放代码对照;
按倒序再解一遍, 核对常驻解码器跨 GOP 复用(不 flush)时取到的仍是同一帧。
"""
import itertools
import os
from unittest.mock import patch

import numpy as np
from openpi_client import image_tools
from PIL import Image

from openpi.policies import teleavatar_v2_policy
import openpi.training.config_lingyu as _config

from openpi.training.lingyu_dataloader_v2.utils.mcap_message_fetcher import MCAP_Message_Fetcher
from openpi.training.lingyu_dataloader_v2.utils.mcap_player import MCAP_Player
from openpi.training.lingyu_dataloader_v2.utils import search_mcap_on_s3
from openpi.training.lingyu_dataloader_v2.utils.search_mcap_on_s3 import find_mcap_url_and_prompt_pairs
from openpi.training.lingyu_dataloader_v2.utils.video_frame_decoder import (
    cpu_decode_current_frame, gpu_decode_current_frames, start_gpu_decoders)
from openpi.training.lingyu_dataloader_v2.lingyu_dataset_v2 import (
    VIDEO_TOPIC_TO_KEY, load_video_decode_config)
from openpi.training.lingyu_dataloader_v2.mcap_config.config import load_mcap_video_topics_gop
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger

logger = get_logger(__file__)

_TESTED_P_COUNTS = (0, 5, 10, 15)  # 待测的 P 帧数: 0 为纯 IDR, 其余需回溯解码
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "video_frame_decoder_output")
# GPU 与 CPU+PIL 结果的平均绝对误差上限: 两种解码器的色度上采样与缩放滤波不同, 实测约 1.8
_MAX_GPU_CPU_MAD = 3.0
_CONFIG_NAME = "pi0_teleavatar_v2_lingyu"  # 裁剪/缩放参数取自这个训练配置


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


def test_gpu():
    """三个视频 topic 在 num_P = 0/5/10/15、旋转与否下, GPU 解出的当前帧与 openpi 的 CPU 处理一致。

    裁剪/缩放参数与训练一样由 load_video_decode_config 从 openpi 配置里取; 对照组走 openpi 的原代码:
    CPU 解码整帧 -> TeleavatarInputs._extract_stereo_view(先转 180° 再裁单目) -> ResizeImages 的 resize_with_pad,
    GPU 结果同样经 resize_with_pad(只补黑边)后再比。
    """
    # 只要一个 mcap 路径, 用不到 prompt, 故跳过翻译(翻译要另配 LINGYU_* 环境变量)
    with patch.object(search_mcap_on_s3, "translate_prompts", lambda prompts, max_workers: prompts):
        path = next(iter(find_mcap_url_and_prompt_pairs()))
    train_config = _config.get_config(_CONFIG_NAME)
    video_key_to_crop, image_resolution = load_video_decode_config(
        train_config.data.create(train_config.assets_dirs, train_config.model))
    video_topics = set(load_mcap_video_topics_gop())
    fetcher = MCAP_Message_Fetcher()
    gpu_decoder_address = start_gpu_decoders()[0]
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    gop_packets = {}  # {(topic_name, num_P): packets}
    for topic_name, msg_type, msg_def, log_time, locations in MCAP_Player(path).play_messages():
        num_p = len(locations) - 1
        if topic_name in video_topics and num_p in _TESTED_P_COUNTS and (topic_name, num_p) not in gop_packets:
            gop_packets[(topic_name, num_p)] = fetcher.fetch_message(msg_type, msg_def, locations)
        if len(gop_packets) == len(video_topics) * len(_TESTED_P_COUNTS):
            break
    assert len(gop_packets) == len(video_topics) * len(_TESTED_P_COUNTS), f"未取全样本: {sorted(gop_packets)}"

    gpu_frames = {}
    # 正序、倒序各解一遍: 解码器跨 GOP 常驻复用, 两遍结果必须逐像素相同; 旋转与否各用一套解码器
    for sample_keys in (sorted(gop_packets), sorted(gop_packets, reverse=True)):
        for (topic_name, num_p), rotate in itertools.product(sample_keys, (False, True)):
            packets = gop_packets[(topic_name, num_p)]
            eye = video_key_to_crop[VIDEO_TOPIC_TO_KEY[topic_name]][0]
            gpu_frame, = gpu_decode_current_frames(gpu_decoder_address, [(
                topic_name, packets[0].encoding, eye, rotate, image_resolution,
                [bytes(packet.data) for packet in packets])])
            if (topic_name, num_p, rotate) in gpu_frames:
                assert np.array_equal(gpu_frames[(topic_name, num_p, rotate)], gpu_frame), \
                    f"{topic_name} num_P={num_p} rotate={rotate} 复用解码器后结果变了"
                continue
            gpu_frames[(topic_name, num_p, rotate)] = gpu_frame

            eye_frame = teleavatar_v2_policy._extract_stereo_view(cpu_decode_current_frame(packets), eye, rotate=rotate)
            cpu_image = image_tools.resize_with_pad(eye_frame, *image_resolution)
            gpu_image = image_tools.resize_with_pad(gpu_frame, *image_resolution)
            assert gpu_image.shape == cpu_image.shape and gpu_image.dtype == np.uint8, \
                f"{topic_name} GPU 输出 {gpu_image.shape} {gpu_image.dtype}, 应为 {cpu_image.shape} uint8"
            mean_abs_diff = np.abs(gpu_image.astype(np.int16) - cpu_image).mean()
            logger.info("%s: num_P=%d, rotate=%s, GPU %s vs CPU+openpi 变换 平均绝对误差 %.2f",
                        topic_name, num_p, rotate, gpu_frame.shape, mean_abs_diff)
            assert mean_abs_diff < _MAX_GPU_CPU_MAD, \
                f"{topic_name} num_P={num_p} rotate={rotate} 误差过大: {mean_abs_diff:.2f}"
            Image.fromarray(gpu_frame).save(os.path.join(
                _OUTPUT_DIR, f"gpu_{topic_name.strip('/').replace('/', '_')}_P{num_p:02d}_rot{int(rotate)}.png"))


if __name__ == "__main__":
    test()
