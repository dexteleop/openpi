#!/usr/bin/env python3
"""
Bring-up test for RTPH265VideoInterface.

This file records decoded RTP/H265 split frames to MP4 for validation. The
production interface stays in rtp_video_interface.py and does not own
file-writing logic.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import signal
import threading
import time

import cv2
import numpy as np

try:
    from .rtp_video_interface import RTPH265VideoInterface
except ImportError:
    from rtp_video_interface import RTPH265VideoInterface


class SplitMp4Recorder:
    """Write split interface images to MP4 without coupling to the decoder."""

    def __init__(
        self,
        *,
        split_output_dir: Path,
        output_fps: float,
    ):
        self.split_output_dir = split_output_dir.expanduser()
        self.output_fps = output_fps
        self.split_output_dir.mkdir(parents=True, exist_ok=True)

        self._writers: dict[str, cv2.VideoWriter] = {}
        self._writer_paths: dict[str, Path] = {}
        self._writer_sizes: dict[str, tuple[int, int]] = {}
        self._frame_counts: dict[str, int] = {}
        self._last_timestamps: dict[str, float] = {}

    def write_latest(self, interface: RTPH265VideoInterface) -> None:
        images, timestamps = interface.get_latest_images_with_timestamps()
        for name in interface.split_image_names:
            self._write_if_new(name, self.split_output_dir / f"{name}.mp4", images, timestamps, name)

    def close(self) -> None:
        for key, writer in self._writers.items():
            writer.release()
            logging.info("Wrote %d frames to %s", self._frame_counts[key], self._writer_paths[key])
        self._writers.clear()

    def _write_if_new(
        self,
        key: str,
        path: Path,
        images: dict[str, np.ndarray],
        timestamps: dict[str, float],
        image_name: str,
    ) -> None:
        frame_rgb = images.get(image_name)
        timestamp = timestamps.get(image_name)
        if frame_rgb is None or timestamp is None:
            return
        if self._last_timestamps.get(key) == timestamp:
            return

        writer = self._get_writer(key, path, frame_rgb)
        size = (frame_rgb.shape[1], frame_rgb.shape[0])
        if size != self._writer_sizes[key]:
            logging.warning("Skipping %s frame with changed size: %s != %s", key, size, self._writer_sizes[key])
            return

        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        self._frame_counts[key] += 1
        self._last_timestamps[key] = timestamp

    def _get_writer(self, key: str, path: Path, frame_rgb: np.ndarray) -> cv2.VideoWriter:
        writer = self._writers.get(key)
        if writer is not None:
            return writer

        size = (frame_rgb.shape[1], frame_rgb.shape[0])
        writer = _open_mp4_writer(path, size, self.output_fps)
        self._writers[key] = writer
        self._writer_paths[key] = path
        self._writer_sizes[key] = size
        self._frame_counts[key] = 0
        logging.info("Recording %s to %s at %.2f fps", key, path, self.output_fps)
        return writer


def _open_mp4_writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer: {path}")
    return writer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--payload", type=int, default=96)
    parser.add_argument("--udp-buffer-size", type=int, default=20_000_000)
    parser.add_argument("--decoder", default="nvh265dec max-display-delay=0")
    parser.add_argument("--log-interval-s", type=float, default=2.0)
    parser.add_argument("--initial-timeout-s", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, help="Record for this many seconds after the first frame")
    # Default matches the ~45 fps RTP stream.
    parser.add_argument("--output-fps", type=float, default=45.0)
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        required=True,
        help="Write one MP4 per split crop to this directory",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    interface = RTPH265VideoInterface(
        camera_name=args.camera_name,
        port=args.port,
        payload=args.payload,
        udp_buffer_size=args.udp_buffer_size,
        decoder=args.decoder,
        log_interval_s=args.log_interval_s,
    )
    recorder = SplitMp4Recorder(
        split_output_dir=args.split_output_dir,
        output_fps=args.output_fps,
    )

    stop_requested = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: stop_requested.set())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_requested.set())

    if args.duration_s is not None:
        logging.info("Will record for %.1fs after the first decoded frame", args.duration_s)

    exit_code = 0
    interface.start()
    try:
        if not interface.wait_for_initial_data(timeout=args.initial_timeout_s):
            return 1

        start_time = time.monotonic()
        while not stop_requested.is_set():
            recorder.write_latest(interface)

            if args.duration_s is not None and time.monotonic() - start_time >= args.duration_s:
                break

            if interface.wait(timeout=0.002):
                exit_code = 1
                break
    finally:
        recorder.close()
        interface.stop()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
