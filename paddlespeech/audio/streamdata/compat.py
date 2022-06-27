# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# See the LICENSE file for licensing terms (BSD-style).
# Modified from https://github.com/webdataset/webdataset
from dataclasses import dataclass
from itertools import islice
from typing import List

import braceexpand, yaml

from . import autodecode
from . import cache, filters, shardlists, tariterators
from .filters import reraise_exception
from .pipeline import DataPipeline
from .paddle_utils import DataLoader, IterableDataset


class FluidInterface:
    def batched(self, batchsize):
        return self.compose(filters.batched(batchsize))

    def dynamic_batched(self, max_frames_in_batch):
        return self.compose(filter.dynamic_batched(max_frames_in_batch))

    def unbatched(self):
        return self.compose(filters.unbatched())

    def listed(self, batchsize, partial=True):
        return self.compose(filters.batched(), batchsize=batchsize, collation_fn=None)

    def unlisted(self):
        return self.compose(filters.unlisted())

    def log_keys(self, logfile=None):
        return self.compose(filters.log_keys(logfile))

    def shuffle(self, size, **kw):
        if size < 1:
            return self
        else:
            return self.compose(filters.shuffle(size, **kw))

    def map(self, f, handler=reraise_exception):
        return self.compose(filters.map(f, handler=handler))

    def decode(self, *args, pre=None, post=None, only=None, partial=False, handler=reraise_exception):
        handlers = [autodecode.ImageHandler(x) if isinstance(x, str) else x for x in args]
        decoder = autodecode.Decoder(handlers, pre=pre, post=post, only=only, partial=partial)
        return self.map(decoder, handler=handler)

    def map_dict(self, handler=reraise_exception, **kw):
        return self.compose(filters.map_dict(handler=handler, **kw))

    def select(self, predicate, **kw):
        return self.compose(filters.select(predicate, **kw))

    def to_tuple(self, *args, handler=reraise_exception):
        return self.compose(filters.to_tuple(*args, handler=handler))

    def map_tuple(self, *args, handler=reraise_exception):
        return self.compose(filters.map_tuple(*args, handler=handler))

    def slice(self, *args):
        return self.compose(filters.slice(*args))

    def rename(self, **kw):
        return self.compose(filters.rename(**kw))

    def rsample(self, p=0.5):
        return self.compose(filters.rsample(p))

    def rename_keys(self, *args, **kw):
        return self.compose(filters.rename_keys(*args, **kw))

    def extract_keys(self, *args, **kw):
        return self.compose(filters.extract_keys(*args, **kw))

    def xdecode(self, *args, **kw):
        return self.compose(filters.xdecode(*args, **kw))

    def data_filter(self, *args, **kw):
        return self.compose(filters.data_filter(*args, **kw))
    
    def tokenize(self, *args, **kw):
        return self.compose(filters.tokenize(*args, **kw))

    def resample(self, *args, **kw):
        return self.compose(filters.resample(*args, **kw)) 

    def compute_fbank(self, *args, **kw):
        return self.compose(filters.compute_fbank(*args, **kw))

    def spec_aug(self, *args, **kw):
        return self.compose(filters.spec_aug(*args, **kw))

    def sort(self, size=500):
        return self.compose(filters.sort(size))

    def padding(self):
        return self.compose(filters.padding())

    def cmvn(self, cmvn_file):
        return self.compose(filters.cmvn(cmvn_file))

class WebDataset(DataPipeline, FluidInterface):
    """Small fluid-interface wrapper for DataPipeline."""

    def __init__(
        self,
        urls,
        handler=reraise_exception,
        resampled=False,
        repeat=False,
        shardshuffle=None,
        cache_size=0,
        cache_dir=None,
        detshuffle=False,
        nodesplitter=shardlists.single_node_only,
        verbose=False,
    ):
        super().__init__()
        if isinstance(urls, IterableDataset):
            assert not resampled
            self.append(urls)
        elif isinstance(urls, str) and (urls.endswith(".yaml") or urls.endswith(".yml")):
            with (open(urls)) as stream:
                spec = yaml.safe_load(stream)
            assert "datasets" in spec
            self.append(shardlists.MultiShardSample(spec))
        elif isinstance(urls, dict):
            assert "datasets" in urls
            self.append(shardlists.MultiShardSample(urls))
        elif resampled:
            self.append(shardlists.ResampledShards(urls))
        else:
            self.append(shardlists.SimpleShardList(urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle is True:
                shardshuffle = 100
            if shardshuffle is not None:
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle))
                else:
                    self.append(filters.shuffle(shardshuffle))
        if cache_size == 0:
            self.append(tariterators.tarfile_to_samples(handler=handler))
        else:
            assert cache_size == -1 or cache_size > 0
            self.append(
                cache.cached_tarfile_to_samples(
                    handler=handler,
                    verbose=verbose,
                    cache_size=cache_size,
                    cache_dir=cache_dir,
                )
            )


class FluidWrapper(DataPipeline, FluidInterface):
    """Small fluid-interface wrapper for DataPipeline."""

    def __init__(self, initial):
        super().__init__()
        self.append(initial)


class WebLoader(DataPipeline, FluidInterface):
    def __init__(self, *args, **kw):
        super().__init__(DataLoader(*args, **kw))
