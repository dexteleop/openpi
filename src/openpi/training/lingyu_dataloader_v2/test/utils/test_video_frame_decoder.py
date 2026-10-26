"""
Play an mcap with MCAP_Player and take num_P = 0/5/10/15 messages from each of the three video topics
(num_P = len(locations) - 1, the number of P frames to decode before the current frame),
fetch the packets by locations with MCAP_Message_Fetcher, hand them to cpu_decode_current_frame() to decode the current frame,
and save the images as local pngs.

test_gpu: the same packets go through the GPU decode process (decoding, mono-eye cropping with rotation, scaling) and are compared against CPU decoding + openpi's original crop and scale code;
decoding again in reverse order verifies that the resident decoder reused across GOPs (no flush) still yields the same frame.
"""
import itertools
import os

import numpy as np
from openpi_client import image_tools
from PIL import Image

from openpi.policies import teleavatar_v2_policy
import openpi.training.config as _config

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

_TESTED_P_COUNTS = (0, 5, 10, 15)  # P-frame counts under test: 0 is pure IDR, the others require tracing back to decode
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "video_frame_decoder_output")
# Upper bound on mean absolute error between GPU and CPU+PIL results: the two decoders differ in chroma upsampling and scaling filters, measured about 1.8
_MAX_GPU_CPU_MAD = 3.0
_CONFIG_NAME = "pi0_teleavatar_v2_lingyu"  # Crop/scale parameters come from this training config


def test():
    """All three video topics decode the current frame at num_P = 0/5/10/15 and save it as png."""
    path = next(iter(find_mcap_url_and_prompt_pairs()))
    video_topics = set(load_mcap_video_topics_gop())
    fetcher = MCAP_Message_Fetcher()
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    logger.info("播放 %s, 视频 topic: %s, 待测 num_P: %s",
                path, sorted(video_topics), _TESTED_P_COUNTS)

    decoded_samples_idx = set()  # Already-decoded (topic_name, num_P) pairs
    wanted_total = len(video_topics) * len(_TESTED_P_COUNTS)
    for message in MCAP_Player(path).play_messages():
        topic_name, msg_type, msg_def, log_time, locations = message
        num_p = len(locations) - 1  # The first location is the IDR, the rest are P frames

        # Only take num_P values of the video topic not yet tested
        if (topic_name not in video_topics
            or num_p not in _TESTED_P_COUNTS
            or (topic_name, num_p) in decoded_samples_idx):
            continue

        # Check that an image can be decoded correctly
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
    """At num_P = 0/5/10/15, with and without rotation, the GPU-decoded current frame matches openpi's CPU processing for all three video topics.

    CPU-decode the whole frame -> TeleavatarInputs._extract_stereo_view (rotate 180° then crop the mono eye) -> ResizeImages's resize_with_pad,
    and the GPU result is likewise put through resize_with_pad (only black-border padding) before comparing.
    Crop/scale parameters come from the openpi config via load_video_decode_config, same as training; the control group runs openpi's original code:
    """
    # Only one mcap path is needed; the prompt is unused
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
    # Decode once forward and once in reverse: the decoder persists across GOPs, so both passes must match pixel-for-pixel; each rotation mode uses its own decoder set
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
