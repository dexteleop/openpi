#!/usr/bin/env python3
"""
Random-access torch Dataset over the WebDataset .tar corpus built by
generate_tars.py.

generate_tars.py writes, next to the shards, a sample_index.json that maps
every sample id to the shard it lives in:

    "0":   {"tar_file": "teleavatar_000000.tar", 
            "shard_index": 0,
            "sample_index_in_shard": 0}

That index is what makes indexed access possible: __init__ reads it once, so
the dataset knows the total sample count and, for any i, which tar to open --
no shard scan, no sequential iteration. This is the difference from the v1
loader, which streams shards through a webdataset pipeline and therefore only
supports iteration.

Each sample in a shard is a pair of members:
  - {id:06d}.json               video paths + per-topic frame indices
  - {id:06d}.state_action.npz   state (1, 62) + action (action_seq_len, 62)

Images are NOT stored in the tars; the json only records which mp4 and which
frame, so __getitem__ decodes that one frame on the fly with torchcodec,
addressing it by frame index directly (see decode_frame).

Decoders are built once per process and kept for the whole run -- see
preload_decoders. Under multiprocessing start method "spawn" they MUST be built
inside the worker: a VideoDecoder pickles to a stub that reports metadata but
raises "Provided stream index=... was not previously added" on the first fetch,
so passing one from the parent yields a decoder that looks alive and is not.
That is why the cache below is a module-level dict rather than an attribute of
the dataset, and why the dataset drops it in __getstate__.

__getitem__ returns:
    {"observation": {"images": {"left_color":  (1, C, H, W) float32 [0,1],
                                "right_color": (1, C, H, W),
                                "head_camera": (1, C, H, W)},
                     "state": (1, 62) float32},
     "action": (action_seq_len, 62) float32}
Frames are channel-first float32 scaled to [0,1]; state / action stay raw as
packed -- no normalization, no unit conversion, no dim selection here.

Wiring it into a DataLoader with spawn workers:

    dataset = TeleavatarTarDataset(wds_dir, preload=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=..., num_workers=N,
        multiprocessing_context="spawn",
        worker_init_fn=worker_init_preload,
        persistent_workers=True,   # required, or every epoch rebuilds
    )

Usage:
    python webdataset_load_tar.py <wds_dir> --num_items N
    python webdataset_load_tar.py /DATA/disk0/haoran/infra_wds --num_items 3
"""

import os
import io
import sys
import json
import time
import logging
import argparse
import tarfile

import numpy as np
import torch

from torchcodec.decoders import VideoDecoder

# Both written by generate_tars.py into the same directory as the shards.
INDEX_FILENAME = "sample_index.json"
VIDEO_INDEX_FILENAME = "video_index.json"

# Recorded topic -> key used in the returned observation["images"] dict.
VIDEO_TOPIC_TO_KEY = {
    "/left/color/image_raw/ffmpeg": "left_color",
    "/right/color/image_raw/ffmpeg": "right_color",
    "/xr_video_topic/ffmpeg": "head_camera",
}

logger = logging.getLogger(__name__)


def npz_decoder(key, data):
    """Decode .npz files from tars into numpy dict."""
    if not key.endswith(".npz"):
        return None
    return dict(np.load(io.BytesIO(data), allow_pickle=True))


def json_decoder(key, data):
    """Decode .json files from tars into dict."""
    if not key.endswith(".json"):
        return None
    return json.loads(data.decode("utf-8"))


def load_sample_index(dataset_dir):
    """Load sample_index.json as a list of (sample_id, entry), sorted by id.

    Returning a list rather than the raw dict fixes the mapping from dataset
    position to sample id: position i is the i-th smallest sample id. For a
    corpus written in one run the ids are 0..n-1, so position == id, but the
    list keeps __getitem__ correct even if a corpus ever carries gaps.
    """
    path = os.path.join(dataset_dir, INDEX_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{INDEX_FILENAME} not found in {dataset_dir}. It is written by "
            f"generate_tars.py; re-run that script to produce it."
        )
    with open(path) as f:
        raw = json.load(f)
    return [(int(k), raw[k]) for k in sorted(raw, key=int)]


def load_video_index(dataset_dir):
    """Load video_index.json as {"<dataset_dir>|<topic>": mp4_path}.

    Returns an empty dict when the file is absent rather than raising: a corpus
    packed before generate_tars.py started writing this index still loads, it
    just builds decoders on first use instead of up front.
    """
    path = os.path.join(dataset_dir, VIDEO_INDEX_FILENAME)
    if not os.path.exists(path):
        logger.warning(
            "%s not found in %s; video decoders will be built on first use "
            "instead of preloaded. Re-run generate_tars.py to write it.",
            VIDEO_INDEX_FILENAME, dataset_dir,
        )
        return {}
    with open(path) as f:
        return json.load(f)


# --- tar handles ----------------------------------------------------------
def _get_tar(pid, tar_path):
    return tarfile.open(tar_path, "r")


def read_sample_from_tar(tar_path, sample_key):
    """Read one sample's (video_meta, state_action) pair out of a shard.

    sample_key is the zero-padded id used by generate_tars.py as the member
    basename, so both members are addressed by name -- no scan of the shard.
    """
    tf = _get_tar(os.getpid(), tar_path)
    json_name = f"{sample_key}.json"
    npz_name = f"{sample_key}.state_action.npz"
    try:
        meta = json_decoder(json_name, tf.extractfile(json_name).read())
        npz = npz_decoder(npz_name, tf.extractfile(npz_name).read())
    except KeyError as e:
        raise KeyError(
            f"{e} missing from {tar_path}; sample_index.json and the shards "
            f"are out of sync (regenerate both with generate_tars.py)."
        ) from e
    return meta, npz


def get_mp4_path(dataset_dir, topic_name):
    """Recorded topic -> the mp4 the extractor wrote for it.

    Same sanitization as the recorder: strip the leading slash and turn the
    remaining slashes into underscores.
    """
    sanitized = topic_name.strip("/").replace("/", "_")
    return os.path.join(dataset_dir, sanitized + ".mp4")


# "exact" rather than torchcodec's cheaper "approximate": approximate seeking
# scales the index by average_fps to guess a position, which lands on the wrong
# frame on this corpus (measured against sequentially-decoded ground truth: max
# pixel diff 0.48 / 0.16 / 0.78 on left / right / head). With seek_mode="exact"
# torchcodec builds a real frame index and get_frames_at(indices=[i]) returned
# the bit-identical frame on all three streams.
SEEK_MODE = "exact"


# path -> VideoDecoder, per process, for the lifetime of the process.
#
# Deliberately module-level rather than an attribute of the dataset: with
# start method "spawn" the dataset is pickled into each worker, and a pickled
# VideoDecoder is a broken stub (see the module docstring). A module-level dict
# is created empty in whichever process imports this module, so each worker
# fills in its own and nothing decoder-shaped ever crosses a process boundary.
_DECODERS = {}

# Building every decoder costs ~0.14s each and ~4 MB resident, so a corpus far
# larger than the one this was written for would make eager preloading expensive
# at worker startup. Warn rather than refuse: the threshold is a hint, not a
# limit.
PRELOAD_WARN_THRESHOLD = int(os.environ.get("VIDEO_PRELOAD_WARN", "300"))


def get_decoder(mp4_path):
    """Return this process's VideoDecoder for one mp4, building it if needed.

    After preload_decoders has run this is a plain dict lookup. The build
    fallback covers the cases preloading cannot: a corpus with no
    video_index.json, or an mp4 referenced by a sample but absent from the
    index.

    seek_mode="exact" is what makes the build expensive -- it scans the
    container to construct a real frame index -- and also what makes frame
    lookup correct, so the decoder is kept rather than rebuilt per access.
    """
    decoder = _DECODERS.get(mp4_path)
    if decoder is None:
        # Announce it: reaching here after preload means a sample referenced an
        # mp4 the index did not cover, and the build stalls this sample.
        t0 = time.perf_counter()
        decoder = VideoDecoder(mp4_path, device="cpu", seek_mode=SEEK_MODE)
        _DECODERS[mp4_path] = decoder
        print(f"[{worker_label()}] on-demand container build "
              f"{os.path.basename(mp4_path)} in "
              f"{time.perf_counter() - t0:.2f}s", flush=True)
    return decoder


def worker_label():
    """Identify the current process for progress lines: "worker N" or "main".

    get_worker_info() returns None in the parent process, which is also the
    num_workers=0 case, so the label distinguishes a real DataLoader worker
    from a preload happening in the main process.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return f"main pid={os.getpid()}"
    return f"worker {info.id}/{info.num_workers} pid={os.getpid()}"


def preload_decoders(video_index, progress=True):
    """Build a decoder for every mp4 in video_index, in THIS process.

    Call this once per process that will decode: from a DataLoader's
    worker_init_fn when num_workers > 0, or directly at dataset construction
    when num_workers == 0. Afterwards no __getitem__ pays a build cost, which
    keeps per-sample latency flat instead of stalling on whichever sample
    happens to touch a given mp4 first (measured: up to 421ms for the longest
    video in this corpus).

    Progress goes to stdout via print rather than the logger on purpose: with
    start method "spawn" the worker is a fresh interpreter that never ran the
    parent's logging.basicConfig, so logger.info() inside a worker is dropped by
    the root logger's default level and the build would look like a silent hang.
    print reaches the terminal from any process.

    Already-built paths are skipped, so calling it twice is harmless.
    """
    paths = sorted(set(video_index.values()))
    label = worker_label()
    if len(paths) > PRELOAD_WARN_THRESHOLD:
        logger.warning(
            "preloading %d video decoders in %s; at ~4MB and ~0.14s each "
            "that is roughly %.1fGB and %.0fs per worker. Set "
            "VIDEO_PRELOAD=lazy to build them on first use instead.",
            len(paths), label, len(paths) * 4 / 1024, len(paths) * 0.14,
        )

    pending = [p for p in paths if p not in _DECODERS]
    if not pending:
        return 0
    if progress:
        print(f"[{label}] building {len(pending)} video container(s)...",
              flush=True)

    t0 = time.perf_counter()
    built = failed = 0
    for n, mp4_path in enumerate(pending, 1):
        t_one = time.perf_counter()
        try:
            _DECODERS[mp4_path] = VideoDecoder(
                mp4_path, device="cpu", seek_mode=SEEK_MODE)
            built += 1
            if progress:
                print(f"[{label}] ({n}/{len(pending)}) "
                      f"{os.path.basename(os.path.dirname(mp4_path))}/"
                      f"{os.path.basename(mp4_path)} "
                      f"built in {time.perf_counter() - t_one:.2f}s",
                      flush=True)
        except Exception as e:  # noqa: BLE001 - a bad video must not kill startup
            # Leave it out of the cache: get_decoder will retry on demand and,
            # if it fails again, _decode_images zero-fills that camera.
            failed += 1
            print(f"[{label}] ({n}/{len(pending)}) FAILED {mp4_path}: {e}",
                  flush=True)

    if progress:
        print(f"[{label}] ready: {built}/{len(pending)} container(s) in "
              f"{time.perf_counter() - t0:.2f}s"
              + (f", {failed} failed" if failed else ""), flush=True)
    return built


def find_video_index(dataset, _depth=0):
    """Dig a video_index out of a possibly-wrapped dataset.

    Training wraps this dataset in TransformedDataset (and potentially more
    layers), each holding the inner one as a private attribute, and none of them
    forward attribute lookups. So a plain getattr(dataset, "video_index") finds
    nothing and preloading silently turns into a no-op -- which looks like
    success until the per-sample stalls show up mid-training. Walk the common
    wrapper attributes instead.
    """
    if dataset is None or _depth > 8:
        return None
    video_index = getattr(dataset, "video_index", None)
    if video_index:
        return video_index
    for attr in ("_dataset", "dataset"):
        inner = getattr(dataset, attr, None)
        if inner is not None and inner is not dataset:
            found = find_video_index(inner, _depth + 1)
            if found:
                return found
    return None


def worker_init_preload(worker_id):  # noqa: ARG001 - signature fixed by torch
    """DataLoader worker_init_fn: preload this worker's decoders.

    Reads the index from the dataset object the worker already holds, so the
    paths travel as plain strings and the decoders are built here, in the
    worker, which is the only place they can be built safely under spawn.
    """
    info = torch.utils.data.get_worker_info()
    video_index = find_video_index(getattr(info, "dataset", None))
    if video_index:
        preload_decoders(video_index)
    else:
        print(f"[{worker_label()}] no video_index reachable from the dataset; "
              f"containers will be built on first use", flush=True)


def get_video_shape(mp4_path):
    """Return (C, H, W) for one mp4, straight from the decoder's metadata.

    Needed so a camera that fails to decode can be zero-filled at its OWN
    resolution; head and wrist differ (3840x1920 vs 2560x800), so borrowing
    another camera's shape would hand back a wrongly-sized frame.
    """
    metadata = get_decoder(mp4_path).metadata
    return (3, metadata.height, metadata.width)


def decode_frame(mp4_path, frame_idx):
    """Decode one frame BY INDEX, returning a (1, C, H, W) float32 tensor.

    The tars record frame indices, and torchcodec addresses frames by index
    natively, so nothing here converts through timestamps. That conversion is
    what made the lerobot path unusable on this corpus: its torchcodec adapter
    only accepts timestamps and recovers an index with round(ts * average_fps),
    which ignores the container's start pts -- these recordings start at a
    non-zero offset that differs per file (left 1.266s, right 0.366s, xr
    1.300s), so the recovered index was tens of frames off.

    Frames come out of torchcodec as uint8; scale to [0,1] float32 to keep the
    layout the transforms downstream already expect.
    """
    frames = get_decoder(mp4_path).get_frames_at(indices=[int(frame_idx)])
    return frames.data.to(torch.float32) / 255.0


class TeleavatarTarDataset(torch.utils.data.Dataset):
    """Indexed Dataset over the tar corpus, driven by sample_index.json.

    __init__ reads the two index files and nothing else: shards stay closed
    until a sample asks for one. The object stays picklable because it holds
    paths only -- open tarfile handles and video decoders are not picklable,
    and a pickled VideoDecoder is worse than unpicklable (it yields a stub that
    raises on first fetch), so both are created inside the process that uses
    them.

    Video decoders are preloaded once per process and kept for the whole run:
      - num_workers == 0: preloaded here in __init__.
      - num_workers > 0:  pass worker_init_preload as the DataLoader's
        worker_init_fn, and set persistent_workers=True so each worker pays
        that cost once per run rather than once per epoch.

    Args:
        dataset_dir: directory holding teleavatar_*.tar, sample_index.json and
            video_index.json.
        preload: build every decoder in this process now. Leave False when the
            DataLoader has workers -- decoders built here cannot reach them
            under spawn, so it would be wasted work in the parent. Defaults to
            the VIDEO_PRELOAD env var ("lazy" disables it).
    """

    def __init__(self, dataset_dir, preload=None):
        self.dataset_dir = dataset_dir
        self.samples = load_sample_index(dataset_dir)
        self.video_index = load_video_index(dataset_dir)
        print(f"Loaded {len(self.samples)} samples from {INDEX_FILENAME} "
              f"in {dataset_dir}")
        if preload is None:
            preload = os.environ.get("VIDEO_PRELOAD", "all") != "lazy"
        if preload and self.video_index:
            preload_decoders(self.video_index)

    def __getstate__(self):
        """Pickle paths and indices only -- never a decoder.

        The decoder cache is module-level, so it already sits outside the
        instance dict and cannot be dragged into a worker by accident. Filtering
        by type here keeps that true even if a decoder is ever assigned to the
        instance, because a pickled VideoDecoder does not fail loudly -- it
        arrives as a stub that raises only on first fetch.
        """
        return {k: v for k, v in self.__dict__.items()
                if not isinstance(v, VideoDecoder)}

    def __len__(self):
        """Number of samples in the corpus, straight from the index."""
        return len(self.samples)

    def _decode_images(self, video_meta):
        """Decode one frame per camera into {key: (1, C, H, W) tensor}.

        A camera that is absent from this sample's json, or whose frame fails
        to decode, is zero-filled at its OWN (1, C, H, W) shape taken from its
        mp4 header -- not at a sibling camera's shape, since head and wrist
        resolutions differ. So the dict always carries all three keys with
        per-camera-consistent shapes and collation never hits a missing entry.
        """
        dataset_path = video_meta["dataset_path"]
        images = {}
        for topic, key in VIDEO_TOPIC_TO_KEY.items():
            mp4_path = get_mp4_path(dataset_path, topic)
            frame_idx = video_meta.get(topic)
            error = None
            if frame_idx is None:
                error = "topic absent from sample"
            else:
                try:
                    images[key] = decode_frame(mp4_path, int(frame_idx))
                    continue
                except Exception as e:
                    error = e
            logger.warning("decode failed (%s frame %s): %s",
                           mp4_path, frame_idx, error)
            images[key] = torch.zeros((1, *get_video_shape(mp4_path)),
                                      dtype=torch.float32)
        return images

    def __getitem__(self, index):
        """Return one sample in the observation/action layout.

        index is a position in the dataset, not a raw sample id; the index
        list maps it to the id, and the id names both the shard member and the
        tar file it lives in.
        """
        sample_id, entry = self.samples[index]
        tar_path = os.path.join(self.dataset_dir, entry["tar_file"])
        video_meta, npz = read_sample_from_tar(tar_path, f"{sample_id:06d}")

        images = self._decode_images(video_meta)
        state = torch.from_numpy(npz["state"].astype(np.float32))
        action = torch.from_numpy(npz["action"].astype(np.float32))

        return {
            "observation": {
                "images": images,
                "state": state,
            },
            "action": action,
        }


def describe_structure(obj, name="item", indent=0):
    """Recursively print type / shape / dtype of every leaf in one sample."""
    pad = "  " * indent
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict(n_keys={len(obj)})")
        for key, value in obj.items():
            describe_structure(value, f"['{key}']", indent + 1)
    elif isinstance(obj, (tuple, list)):
        print(f"{pad}{name}: {type(obj).__name__}(len={len(obj)})")
        for i, value in enumerate(obj):
            describe_structure(value, f"[{i}]", indent + 1)
    elif isinstance(obj, torch.Tensor):
        print(f"{pad}{name}: Tensor shape={tuple(obj.shape)} "
              f"dtype={obj.dtype} min={obj.min():.4f} max={obj.max():.4f}")
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} = {obj!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect the indexed tar Dataset")
    parser.add_argument("dataset_dir", help="Directory with .tar + "
                                            f"{INDEX_FILENAME}")
    parser.add_argument("--num_items", type=int, default=2,
                        help="How many single items to inspect")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Error: '{args.dataset_dir}' is not a directory")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    t_init = time.perf_counter()
    dataset = TeleavatarTarDataset(args.dataset_dir)
    print(f"len(dataset) = {len(dataset)}  "
          f"videos indexed = {len(dataset.video_index)}  "
          f"init (incl. preload) = {time.perf_counter() - t_init:.2f}s\n")

    for idx in range(min(args.num_items, len(dataset))):
        t0 = time.perf_counter()
        item = dataset[idx]
        elapsed = time.perf_counter() - t0
        print(f"--- Item {idx} (loaded in {elapsed:.3f}s) ---")
        describe_structure(item, name="item")
        print()
