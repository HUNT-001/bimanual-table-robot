"""Streaming loader over the shards of policy/shards.py (kept apart so data collection does not import torch)."""
from __future__ import annotations

import os

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from policy.shards import ARRAYS


class ShardStream(IterableDataset):
    """Iterable dataset over shards that never holds more than `buffer_shards` shards in memory.

    Shards are shuffled each epoch, read in groups of `buffer_shards`, and samples are shuffled inside the group.
    With DataLoader workers, each worker takes a disjoint subset of the shards. Yields (index into the group
    arrays, segment end) through `make_sample`, supplied by the training script so the sampling and augmentation
    code stays in one place."""

    def __init__(self, root, rows, make_sample, buffer_shards=16, seed=0):
        self.root, self.rows, self.make_sample = root, rows, make_sample
        self.buffer_shards, self.seed, self.epoch = buffer_shards, seed, 0
        self.n_samples = sum(r["n_frames"] for r in rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng(self.seed + 1000 * self.epoch + wid)
        order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.rows))[wid::nw]
        for g in range(0, len(order), self.buffer_shards):
            group = [self.rows[i] for i in order[g:g + self.buffer_shards]]
            data = [np.load(os.path.join(self.root, r["path"])) for r in group]
            arrs = {k: np.concatenate([d[k] for d in data]) for k in ARRAYS}
            ends, off = np.empty(len(arrs["state"]), dtype=np.int64), 0
            for r in group:                                   # segment end for every frame of the shard
                starts = r["segment_starts"] + [r["n_frames"]]
                for a, b in zip(starts[:-1], starts[1:]):
                    ends[off + a:off + b] = off + b
                off += r["n_frames"]
            for i in rng.permutation(len(arrs["state"])):
                yield self.make_sample(arrs, int(i), int(ends[i]))
