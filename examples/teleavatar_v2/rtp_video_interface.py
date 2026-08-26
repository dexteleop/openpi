#!/usr/bin/env python3
"""
Receive a Teleavatar H265 RTP video stream as RGB images.

This mirrors the image-facing part of TeleavatarROS2Interface: decoded frames
are stored in latest_images under a lock, and get_observation() returns image
copies. The interface keeps both the composite frame and six split camera views.
Test-only file writing lives in test.py.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict
from typing import Optional

import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib
    from gi.repository import Gst
except ImportError as exc:
    raise RuntimeError(
        "PyGObject/GStreamer Python bindings are required. Install python3-gi "
        "and gir1.2-gstreamer-1.0, or run with a Python that has gi available."
    ) from exc


Gst.init(None)


TELEAVATAR_SPLIT_REGIONS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("head_right_eye", (0.1250, 0.0000, 0.8750, 0.3529411765)),
    ("head_left_eye", (0.1250, 0.3529411765, 0.8750, 0.7058823529)),
    ("right_wrist_left_eye", (0.0000, 0.7058823529, 0.5000, 0.8529411765)),
    ("right_wrist_right_eye", (0.5000, 0.7058823529, 1.0000, 0.8529411765)),
    ("left_wrist_left_eye", (0.0000, 0.8529411765, 0.5000, 1.0000)),
    ("left_wrist_right_eye", (0.5000, 0.8529411765, 1.0000, 1.0000)),
)


class RTPH265VideoInterface:
    """Thread-safe image interface backed by a GStreamer RTP/H265 decoder."""

    def __init__(
        self,
        *,
        camera_name: str = "head_camera",
        port: int = 8890,
        payload: int = 96,
        udp_buffer_size: int = 20_000_000,
        rtp_queue_depth_s: float = 2.0,
        rtp_packet_rate_hz: float = 2115.0,
        decoder: str = "nvh265dec max-display-delay=0",
        log_interval_s: float = 2.0,
        watchdog_interval_s: float = 0.1,
        stall_timeout_s: float = 1.0,
        startup_grace_s: float = 3.0,
        restart_backoff_s: float = 2.0,
    ):
        self.camera_name = camera_name
        self.port = port
        self.payload = payload
        self.udp_buffer_size = udp_buffer_size
        # Depth of the pre-decode RTP queue, in seconds of stream. Time-based
        # so it stays correct if the source's bitrate or resolution changes;
        # rtp_packet_rate_hz only sizes the buffer-count and byte safety nets
        # (measured: 45 fps x 47 pkts/frame ~= 2115 pkt/s).
        self.rtp_queue_depth_s = rtp_queue_depth_s
        self.rtp_packet_rate_hz = rtp_packet_rate_hz
        self.decoder = decoder
        self.log_interval_s = log_interval_s
        # Watchdog: no decoded frame for stall_timeout_s is a stall. Whether we
        # restart depends on whether RTP packets are still arriving — see
        # _watchdog_loop. A restart costs ~1-1.5s (state change + wait for the
        # next IDR, which this stream sends exactly every 1.0s), so restarting
        # a merely-paused source makes the outage strictly worse.
        self.watchdog_interval_s = watchdog_interval_s
        self.stall_timeout_s = stall_timeout_s
        # Before the first frame of a pipeline instance a longer budget applies:
        # startup has to negotiate caps and then wait for the next IDR, which
        # this stream only sends every 1.0s. Using stall_timeout_s here would
        # make the watchdog restart the pipeline during normal startup.
        self.startup_grace_s = startup_grace_s
        self.restart_backoff_s = restart_backoff_s
        self.split_regions = TELEAVATAR_SPLIT_REGIONS
        self.split_image_names = tuple(name for name, _box in self.split_regions)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.lock = threading.Lock()
        self.latest_images: Dict[str, np.ndarray] = {}
        self.image_timestamps: Dict[str, float] = {}

        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._first_frame = threading.Event()
        self._eos_or_error = threading.Event()
        self._stop_requested = False

        self._frame_count = 0
        self._rtp_packet_count = 0
        # _rtp_packet_count is per-pipeline-instance and resets on restart;
        # this carries the total across restarts so the watchdog's "is the
        # source still sending?" test survives one.
        self._rtp_packets_base = 0
        self._h265_buffer_count = 0
        self._stats_start_time: Optional[float] = None
        self._last_log_time = time.monotonic()
        self._last_log_frame_count = 0
        # PTS-keyed (not a FIFO deque) so a frame dropped at the leaky appsink can't desync pairing.
        # Written on the udpsrc streaming thread, popped on the appsink thread -> needs its own lock.
        self._decode_lock = threading.Lock()
        self._decode_start_times: dict[int, float] = {}
        self._latest_decode_latency_ms: Optional[float] = None

        # Watchdog state. _restart_lock serializes stop()/start() between the
        # watchdog thread and destroy_node().
        self._restart_lock = threading.RLock()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        self._last_frame_monotonic: Optional[float] = None
        self._pipeline_start_monotonic: Optional[float] = None
        self._restart_count = 0
        self._outage_start: Optional[float] = None
        self._outage_start_packets = 0
        self._outage_silent_ticks = 0
        self._outage_moving_ticks = 0
        self._outage_count = 0

    def start(self) -> None:
        """Start receiving images in a background GStreamer thread, plus the watchdog."""
        with self._restart_lock:
            self._start_pipeline()

        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_stop.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name="rtp-h265-watchdog", daemon=True
            )
            self._watchdog_thread.start()

    def _start_pipeline(self) -> None:
        """Build and run the GStreamer pipeline. Caller must hold _restart_lock."""
        if self._thread and self._thread.is_alive():
            return

        self._reset_runtime_state()
        pipeline_description = self._build_pipeline()
        self.logger.info("Starting GStreamer pipeline: %s", pipeline_description)

        pipeline = Gst.parse_launch(pipeline_description)
        if not isinstance(pipeline, Gst.Pipeline):
            raise RuntimeError("GStreamer description did not create a Pipeline")

        appsink = pipeline.get_by_name("decoded_frames")
        if appsink is None:
            raise RuntimeError("Failed to find appsink named decoded_frames")
        appsink.connect("new-sample", self._image_callback)

        rtp_probe = pipeline.get_by_name("rtp_probe")
        if rtp_probe is not None:
            rtp_probe.connect("handoff", self._on_rtp_packet)

        h265_probe = pipeline.get_by_name("h265_probe")
        if h265_probe is not None:
            h265_probe.connect("handoff", self._on_h265_buffer)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._pipeline = pipeline
        self._loop = GLib.MainLoop()
        self._first_frame.clear()
        self._eos_or_error.clear()
        self._stop_requested = False

        self._thread = threading.Thread(target=self._run_loop, name="rtp-h265-video", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the watchdog and the GStreamer pipeline."""
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=1.0)
            self._watchdog_thread = None

        with self._restart_lock:
            self._stop_pipeline()
        # _frame_count is per-pipeline-instance, so report the restart-safe total.
        self.logger.info(
            "Stopped decoder after %d frames (%d pipeline restarts, %d outages)",
            self._frame_count,
            self._restart_count,
            self._outage_count,
        )

    def _stop_pipeline(self) -> None:
        """Tear the pipeline down. Caller must hold _restart_lock."""
        self._stop_requested = True

        if self._thread is not None:
            # quit() is a no-op on a loop that has not started running yet,
            # so retry until the loop thread actually exits (bounded).
            deadline = time.monotonic() + 2.0
            while self._thread.is_alive() and time.monotonic() < deadline:
                if self._loop is not None and self._loop.is_running():
                    self._loop.quit()
                self._thread.join(timeout=0.1)
            if self._thread.is_alive():
                self.logger.warning("RTP decoder thread did not exit within 2s")
            self._thread = None

        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

        self._loop = None

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        """Wait for the first decoded image, matching TeleavatarROS2Interface."""
        self.logger.info("Waiting for initial video frame...")
        if self._first_frame.wait(timeout=timeout):
            self.logger.info("Initial video frame received")
            return True

        self.logger.error(
            "Timeout waiting for video after %.1fs: rtp_packets=%d h265_buffers=%d",
            timeout,
            self._rtp_packet_count,
            self._h265_buffer_count,
        )
        return False

    def get_observation(self) -> Optional[dict]:
        """Return current image observation, or None before the first frame."""
        with self.lock:
            if self.camera_name not in self.latest_images:
                return None
            return {"images": {name: image.copy() for name, image in self.latest_images.items()}}

    def get_latest_image(self) -> Optional[np.ndarray]:
        """Return the latest composite RGB image."""
        with self.lock:
            image = self.latest_images.get(self.camera_name)
            return None if image is None else image.copy()

    def get_latest_images(self) -> Dict[str, np.ndarray]:
        """Return all latest images, including the composite and split crops."""
        with self.lock:
            return {name: image.copy() for name, image in self.latest_images.items()}

    def get_latest_images_with_timestamps(self) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Return latest image copies and their receive timestamps."""
        with self.lock:
            images = {name: image.copy() for name, image in self.latest_images.items()}
            return images, dict(self.image_timestamps)

    def has_initial_frame(self) -> bool:
        return self._first_frame.is_set()

    def stream_ended(self) -> bool:
        """True once the pipeline hit EOS or an unrecoverable error (no new frames will arrive)."""
        return self._eos_or_error.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for EOS/error. Returns True if the stream ended."""
        return self._eos_or_error.wait(timeout=timeout)

    def _watchdog_loop(self) -> None:
        """Detect video stalls and recover from the ones that are ours to fix.

        The classification matters more than the detection. A pipeline restart
        costs ~1-1.5s (NULL->PLAYING plus waiting for the next IDR, which this
        stream emits exactly every 1.0s). So restarting is only right when the
        receiver is wedged; if the *source* stopped, a restart adds blind time
        and fixes nothing. The two are told apart by whether RTP packets are
        still arriving while frames are not.
        """
        last_packets = self._total_rtp_packets()
        last_restart = 0.0
        last_silent_log = 0.0

        while not self._watchdog_stop.wait(self.watchdog_interval_s):
            now = time.monotonic()
            packets = self._total_rtp_packets()
            packets_moving = packets != last_packets
            last_packets = packets

            start_t = self._pipeline_start_monotonic
            if start_t is None:
                continue
            frame_t = self._last_frame_monotonic
            # A frame from a *previous* pipeline instance doesn't prove this one
            # is alive, so it only counts if it arrived after this start.
            have_frame_this_run = frame_t is not None and frame_t >= start_t
            reference = frame_t if have_frame_this_run else start_t
            timeout = self.stall_timeout_s if have_frame_this_run else self.startup_grace_s

            since_frame = now - reference
            errored = self._eos_or_error.is_set()

            if since_frame <= timeout and not errored:
                self._end_outage(now, packets)
                continue

            self._begin_outage(now, packets, packets_moving)
            if packets_moving:
                self._outage_moving_ticks += 1
            else:
                self._outage_silent_ticks += 1

            if errored:
                reason = "pipeline reported EOS/ERROR"
            elif packets_moving:
                reason = (
                    f"receiver wedged — no decoded frame for {since_frame:.2f}s "
                    "but RTP packets are still arriving"
                )
            else:
                if now - last_silent_log >= 2.0:
                    last_silent_log = now
                    self.logger.error(
                        "RTP video outage: no frame for %.2fs and no RTP packets arriving — "
                        "the source stopped sending. NOT restarting the pipeline (a restart "
                        "costs ~1-1.5s of extra blind time and cannot conjure frames); "
                        "waiting for the sender to resume.",
                        since_frame,
                    )
                continue

            if now - last_restart < self.restart_backoff_s:
                continue
            last_restart = now
            self._restart_pipeline(reason)
            last_packets = self._total_rtp_packets()

    def _total_rtp_packets(self) -> int:
        """RTP packets seen across all pipeline instances (restart-safe)."""
        return self._rtp_packets_base + self._rtp_packet_count

    def _begin_outage(self, now: float, packets: int, packets_moving: bool) -> None:
        if self._outage_start is not None:
            return
        self._outage_start = now
        self._outage_start_packets = packets
        self._outage_silent_ticks = 0
        self._outage_moving_ticks = 0
        self._outage_count += 1
        self.logger.error(
            "RTP video outage #%d started (rtp_packets=%d, source %s)",
            self._outage_count,
            packets,
            "still sending" if packets_moving else "silent",
        )

    def _end_outage(self, now: float, packets: int) -> None:
        if self._outage_start is None:
            return
        duration = now - self._outage_start
        during = packets - self._outage_start_packets
        # Classify on whether the source was ever observed silent, not on the
        # packet delta: after a source outage, packets resume up to 1.0s before
        # the first decodable frame (the IDR wait), so the delta is nonzero even
        # for a pure source stop. A wedged receiver never sees a silent tick.
        kind = "source-stopped" if self._outage_silent_ticks else "receiver-wedge"
        # Structured so long runs can be grepped to characterise how often the
        # S100 actually drops out — that evidence is what drives a real fix,
        # since the source stopping is not something this client can prevent.
        self.logger.warning(
            "RTP_OUTAGE_END #%d duration=%.2fs kind=%s rtp_packets_during=%d "
            "silent_ticks=%d moving_ticks=%d restarts=%d",
            self._outage_count,
            duration,
            kind,
            during,
            self._outage_silent_ticks,
            self._outage_moving_ticks,
            self._restart_count,
        )
        self._outage_start = None

    def _restart_pipeline(self, reason: str) -> None:
        with self._restart_lock:
            # stop() may have run while we were waiting for the lock (it only
            # joins the watchdog thread with a timeout, so this thread can
            # still be in flight). Restarting now would resurrect a pipeline
            # the owner just tore down.
            if self._watchdog_stop.is_set():
                return
            self._restart_count += 1
            self.logger.error(
                "Restarting RTP pipeline (#%d): %s. Expect ~1-1.5s of blind time.",
                self._restart_count,
                reason,
            )
            try:
                self._stop_pipeline()
                self._start_pipeline()
            except Exception:
                self.logger.exception("RTP pipeline restart failed; retrying after backoff")

    def _reset_runtime_state(self) -> None:
        self._rtp_packets_base += self._rtp_packet_count
        self._frame_count = 0
        self._rtp_packet_count = 0
        self._h265_buffer_count = 0
        self._stats_start_time = None
        self._last_log_time = time.monotonic()
        self._last_log_frame_count = 0
        with self._decode_lock:
            self._decode_start_times.clear()
        self._latest_decode_latency_ms = None
        # _last_frame_monotonic is deliberately NOT reset: an in-progress
        # outage is not over until a frame actually arrives.
        self._pipeline_start_monotonic = time.monotonic()

    def _build_pipeline(self) -> str:
        caps = (
            "application/x-rtp,"
            "media=video,"
            "clock-rate=90000,"
            "encoding-name=H265,"
            f"payload={self.payload}"
        )
        # All three limits are armed: whichever is hit first blocks upstream.
        # max-size-time is the one that expresses the intent (N seconds of
        # stream) and is bitrate-independent; the other two are safety nets.
        # They matter because GstQueue has to *estimate* level-time from the
        # first/last buffer timestamps (RTP buffers carry no duration), so a
        # burst or a timestamp discontinuity can make the time estimate lag
        # reality. Without the count/byte caps a stalled decoder could grow
        # this queue unboundedly on a bad estimate; with them the worst case
        # is bounded at ~1.5x the nominal depth in RAM.
        queue_time_ns = int(self.rtp_queue_depth_s * Gst.SECOND)
        # 1.5x headroom so the count cap only fires when the time estimate is
        # wrong, not during normal jitter.
        queue_buffers = int(self.rtp_queue_depth_s * self.rtp_packet_rate_hz * 1.5)
        # MTU-sized packets; generous per-packet allowance.
        queue_bytes = queue_buffers * 1500
        decode = (
            f"udpsrc port={self.port} buffer-size={self.udp_buffer_size} caps=\"{caps}\" "
            # Decouples reading the socket from decoding: without it udpsrc,
            # the probe callback and the whole CUDA chain share one thread, so
            # a GPU stall stops the socket being drained. Non-leaky — dropping
            # RTP packets mid-frame would corrupt the access unit; if this ever
            # fills, the watchdog restarts the pipeline instead.
            f"! queue max-size-buffers={queue_buffers} max-size-bytes={queue_bytes} "
            f"max-size-time={queue_time_ns} "
            "! identity name=rtp_probe signal-handoffs=true silent=true "
            "! rtph265depay "
            "! h265parse config-interval=-1 disable-passthrough=true "
            "! video/x-h265,stream-format=byte-stream,alignment=au "
            "! identity name=h265_probe signal-handoffs=true silent=true "
            f"! {self.decoder} "
            # GPU NV12->RGB in CUDA memory + single download; ~3x lower latency than CPU videoconvert.
            "! cudaconvert "
            "! cudadownload "
            "! video/x-raw,format=RGB "
        )
        appsink = (
            "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 "
            "! appsink name=decoded_frames emit-signals=true max-buffers=1 drop=true sync=false"
        )
        return f"{decode} ! {appsink}"

    def _run_loop(self) -> None:
        if self._pipeline is None or self._loop is None:
            raise RuntimeError("Pipeline was not initialized")

        pipeline = self._pipeline
        loop = self._loop
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.logger.error("Failed to set GStreamer pipeline to PLAYING")
            self._eos_or_error.set()
            return

        try:
            # stop() may have been requested before the loop started running;
            # quit() would have been lost, so check the flag first (stop()
            # also keeps retrying quit() until this thread exits).
            if not self._stop_requested:
                loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL)

    def _image_callback(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        decoded_at = time.monotonic()
        out_buffer = sample.get_buffer()
        out_pts = out_buffer.pts if out_buffer is not None else Gst.CLOCK_TIME_NONE

        try:
            frame = _sample_to_rgb(sample)
        except Exception:
            self.logger.exception("Failed to convert decoded sample to RGB")
            return Gst.FlowReturn.OK

        split_frames = self._split_frame(frame)
        self._update_decode_latency(out_pts, decoded_at)

        timestamp = time.time()
        with self.lock:
            self.latest_images[self.camera_name] = frame
            self.image_timestamps[self.camera_name] = timestamp
            for name, split_frame in split_frames.items():
                self.latest_images[name] = split_frame
                self.image_timestamps[name] = timestamp

        now = time.monotonic()
        self._frame_count += 1
        self._last_frame_monotonic = now
        self._first_frame.set()
        self._log_stats(frame, now)
        return Gst.FlowReturn.OK

    def _on_rtp_packet(self, _identity: Gst.Element, _buffer: Gst.Buffer) -> None:
        self._rtp_packet_count += 1

    def _on_h265_buffer(self, _identity: Gst.Element, buffer: Gst.Buffer) -> None:
        self._h265_buffer_count += 1
        pts = buffer.pts
        if pts != Gst.CLOCK_TIME_NONE:
            # Written here on the udpsrc streaming thread, popped on the appsink
            # thread — without the lock the eviction below can race the pop and
            # raise KeyError inside a GObject signal handler.
            with self._decode_lock:
                self._decode_start_times[pts] = time.monotonic()
                # evict oldest; entries for dropped frames are never popped
                while len(self._decode_start_times) > 512:
                    del self._decode_start_times[next(iter(self._decode_start_times))]

    def _update_decode_latency(self, pts: int, decoded_at: float) -> None:
        with self._decode_lock:
            start = self._decode_start_times.pop(pts, None)
        if start is not None:
            self._latest_decode_latency_ms = max(0.0, (decoded_at - start) * 1000.0)

    def _split_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        height, width = frame.shape[:2]
        crops: Dict[str, np.ndarray] = {}
        for name, box in self.split_regions:
            x1, y1, x2, y2 = _normalized_box_to_pixels(box, width, height)
            if x2 > x1 and y2 > y1:
                crops[name] = frame[y1:y2, x1:x2].copy()
        return crops

    def _log_stats(self, frame: np.ndarray, now: float) -> None:
        if self._stats_start_time is None:
            self._stats_start_time = now
            self._last_log_time = now
            self._last_log_frame_count = self._frame_count
            return

        elapsed = now - self._last_log_time
        if elapsed < self.log_interval_s:
            return

        frames_since_last_log = self._frame_count - self._last_log_frame_count
        interval_fps = frames_since_last_log / elapsed if elapsed > 0.0 else 0.0
        overall_elapsed = now - self._stats_start_time
        overall_fps = (self._frame_count - 1) / overall_elapsed if overall_elapsed > 0.0 else 0.0
        self._last_log_time = now
        self._last_log_frame_count = self._frame_count

        latency = "n/a" if self._latest_decode_latency_ms is None else f"{self._latest_decode_latency_ms:.1f} ms"
        self.logger.info(
            "camera=%s frames=%d rtp_packets=%d h265_buffers=%d shape=%s fps=%.2f "
            "overall_fps=%.2f decode_latency=%s",
            self.camera_name,
            self._frame_count,
            self._rtp_packet_count,
            self._h265_buffer_count,
            tuple(frame.shape),
            interval_fps,
            overall_fps,
            latency,
        )

    def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> None:
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            self.logger.error("GStreamer error from %s: %s", message.src.get_name(), error)
            if debug:
                self.logger.error("GStreamer debug info: %s", debug)
            self._eos_or_error.set()
            if self._loop is not None:
                self._loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            # Not fatal, but the only channel for things like udpsrc failing to
            # get the receive buffer it asked for. Swallowing these is how a
            # 94x-undersized socket buffer went unnoticed.
            warning, debug = message.parse_warning()
            self.logger.warning("GStreamer warning from %s: %s", message.src.get_name(), warning)
            if debug:
                self.logger.warning("GStreamer debug info: %s", debug)
        elif message.type == Gst.MessageType.EOS:
            self._eos_or_error.set()
            if self._loop is not None:
                self._loop.quit()


def _sample_to_rgb(sample: Gst.Sample) -> np.ndarray:
    caps = sample.get_caps()
    if caps is None or caps.get_size() == 0:
        raise RuntimeError("Decoded sample has no caps")

    structure = caps.get_structure(0)
    width = int(structure.get_value("width"))
    height = int(structure.get_value("height"))
    if structure.get_value("format") != "RGB":
        raise RuntimeError(f"Expected RGB sample, got {structure.get_value('format')}")

    buffer = sample.get_buffer()
    if buffer is None:
        raise RuntimeError("Decoded sample has no buffer")

    ok, map_info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("Failed to map decoded frame buffer")

    try:
        row_bytes = width * 3
        raw = np.frombuffer(map_info.data, dtype=np.uint8)
        stride = raw.size // height if raw.size % height == 0 and raw.size // height >= row_bytes else row_bytes
        return raw[: height * stride].reshape((height, stride))[:, :row_bytes].reshape((height, width, 3)).copy()
    finally:
        buffer.unmap(map_info)


def _normalized_box_to_pixels(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, round(x1 * width))),
        max(0, min(height, round(y1 * height))),
        max(0, min(width, round(x2 * width))),
        max(0, min(height, round(y2 * height))),
    )
