#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

# Modified from https://github.com/webdataset/webdataset

"""Train PyTorch models directly from POSIX tar archive.

Code works locally or over HTTP connections.
"""

import os, random, sys, time
from dataclasses import dataclass, field
from itertools import islice
from typing import List

import braceexpand, yaml

from . import utils
from .filters import pipelinefilter
from .paddle_utils import IterableDataset


def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)


class SimpleShardList(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, urls, seed=None):
        """Iterate through the list of shards.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.seed = seed

    def __len__(self):
        return len(self.urls)

    def __iter__(self):
        """Return an iterator over the shards."""
        urls = self.urls.copy()
        if self.seed is not None:
            random.Random(self.seed).shuffle(urls)
        for url in urls:
            yield dict(url=url)


def split_by_node(src, group=None):
    rank, world_size, worker, num_workers = utils.paddle_worker_info(group=group)
    if world_size > 1:
        for s in islice(src, rank, None, world_size):
            yield s
    else:
        for s in src:
            yield s


def single_node_only(src, group=None):
    rank, world_size, worker, num_workers = utils.paddle_worker_info(group=group)
    if world_size > 1:
        raise ValueError("input pipeline needs to be reconfigured for multinode training")
    for s in src:
        yield s


def split_by_worker(src):
    rank, world_size, worker, num_workers = utils.paddle_worker_info()
    if num_workers > 1:
        for s in islice(src, worker, None, num_workers):
            yield s
    else:
        for s in src:
            yield s


def resampled_(src, n=sys.maxsize):
    import random

    seed = time.time()
    try:
        seed = open("/dev/random", "rb").read(20)
    except Exception as exn:
        print(repr(exn)[:50], file=sys.stderr)
    rng = random.Random(seed)
    print("# resampled loading", file=sys.stderr)
    items = list(src)
    print(f"# resampled got {len(items)} samples, yielding {n}", file=sys.stderr)
    for i in range(n):
        yield rng.choice(items)


resampled = pipelinefilter(resampled_)


def non_empty(src):
    count = 0
    for s in src:
        yield s
        count += 1
    if count == 0:
        raise ValueError("pipeline stage received no data at all and this was declared as an error")


@dataclass
class MSSource:
    """Class representing a data source."""

    name: str = ""
    perepoch: int = -1
    resample: bool = False
    urls: List[str] = field(default_factory=list)


default_rng = random.Random()


def expand(s):
    return os.path.expanduser(os.path.expandvars(s))


class MultiShardSample(IterableDataset):
    def __init__(self, fname):
        """Construct a shardlist from multiple sources using a YAML spec."""
        self.epoch = -1
class MultiShardSample(IterableDataset):
    def __init__(self, fname):
        """Construct a shardlist from multiple sources using a YAML spec."""
        self.epoch = -1
        self.parse_spec(fname)

    def parse_spec(self, fname):
        self.rng = default_rng  # capture default_rng if we fork
        if isinstance(fname, dict):
            spec = fname
            fname = "{dict}"
        else:
            with open(fname) as stream:
                spec = yaml.safe_load(stream)
        assert set(spec.keys()).issubset(set("prefix datasets buckets".split())), list(spec.keys())
        prefix = expand(spec.get("prefix", ""))
        self.sources = []
        for ds in spec["datasets"]:
            assert set(ds.keys()).issubset(set("buckets name shards resample choose".split())), list(
                ds.keys()
            )
            buckets = ds.get("buckets", spec.get("buckets", []))
            if isinstance(buckets, str):
                buckets = [buckets]
            buckets = [expand(s) for s in buckets]
            if buckets == []:
                buckets = [""]
            assert len(buckets) == 1, f"{buckets}: FIXME support for multiple buckets unimplemented"
            bucket = buckets[0]
            name = ds.get("name", "@" + bucket)
            urls = ds["shards"]
            if isinstance(urls, str):
                urls = [urls]
            # urls = [u for url in urls for u in braceexpand.braceexpand(url)]
            urls = [
                prefix + os.path.join(bucket, u) for url in urls for u in braceexpand.braceexpand(expand(url))
            ]
            resample = ds.get("resample", -1)
            nsample = ds.get("choose", -1)
            if nsample > len(urls):
                raise ValueError(f"perepoch {nsample} must be no greater than the number of shards")
            if (nsample > 0) and (resample > 0):
                raise ValueError("specify only one of perepoch or choose")
            entry = MSSource(name=name, urls=urls, perepoch=nsample, resample=resample)
            self.sources.append(entry)
            print(f"# {name} {len(urls)} {nsample}", file=sys.stderr)

    def set_epoch(self, seed):
        """Set the current epoch (for consistent shard selection among nodes)."""
        self.rng = random.Random(seed)

    def get_shards_for_epoch(self):
        result = []
        for source in self.sources:
            if source.resample > 0:
                # sample with replacement
                l = self.rng.choices(source.urls, k=source.resample)
            elif source.perepoch > 0:
                # sample without replacement
                l = list(source.urls)
                self.rng.shuffle(l)
                l = l[: source.perepoch]
            else:
                l = list(source.urls)
            result += l
        self.rng.shuffle(result)
        return result

    def __iter__(self):
        shards = self.get_shards_for_epoch()
        for shard in shards:
            yield dict(url=shard)


def shardspec(spec):
    if spec.endswith(".yaml"):
        return MultiShardSample(spec)
    else:
        return SimpleShardList(spec)


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.worker_seed = utils.paddle_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = -1

    def __iter__(self):
        """Return an iterator over the shards."""
        self.epoch += 1
        if self.deterministic:
            seed = utils.make_seed(self.worker_seed(), self.epoch)
        else:
            seed = utils.make_seed(self.worker_seed(), self.epoch, os.getpid(), time.time_ns(), os.urandom(4))
        if os.environ.get("WDS_SHOW_SEED", "0") == "1":
            print(f"# ResampledShards seed {seed}")
        self.rng = random.Random(seed)
        for _ in range(self.nshards):
            index = self.rng.randint(0, len(self.urls) - 1)
            yield dict(url=self.urls[index])
