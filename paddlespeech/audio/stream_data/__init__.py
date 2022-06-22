# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).
# Modified from https://github.com/webdataset/webdataset
#
# flake8: noqa

from .cache import (
    cached_tarfile_samples,
    cached_tarfile_to_samples,
    lru_cleanup,
    pipe_cleaner,
)
from .compat import WebDataset, WebLoader, FluidWrapper
from webdataset.extradatasets import MockDataset, with_epoch, with_length
from .filters import (
    associate,
    batched,
    decode,
    detshuffle,
    extract_keys,
    getfirst,
    info,
    map,
    map_dict,
    map_tuple,
    pipelinefilter,
    rename,
    rename_keys,
    rsample,
    select,
    shuffle,
    slice,
    to_tuple,
    transform_with,
    unbatched,
    xdecode,
    data_filter,
    tokenize,
    resample,
    compute_fbank,
    spec_aug,
    sort,
    padding,
    cmvn
)
from webdataset.handlers import (
    ignore_and_continue,
    ignore_and_stop,
    reraise_exception,
    warn_and_continue,
    warn_and_stop,
)
from .pipeline import DataPipeline
from .shardlists import (
    MultiShardSample,
    ResampledShards,
    SimpleShardList,
    non_empty,
    resampled,
    shardspec,
    single_node_only,
    split_by_node,
    split_by_worker,
)
from .tariterators import tarfile_samples, tarfile_to_samples
from .utils import PipelineStage, repeatedly
from webdataset.writer import ShardWriter, TarWriter, numpy_dumps
from webdataset.mix import RandomMix, RoundRobin
