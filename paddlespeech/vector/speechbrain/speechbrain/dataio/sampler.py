"""PyTorch compatible samplers.

These determine the order of iteration through a dataset.

Authors:
  * Aku Rouhe 2020
  * Samuele Cornell 2020
  * Ralf Leibold 2020
  * Andreas Nautsch 2021
"""
import paddle
import logging
from operator import itemgetter
from paddle.io import (
    RandomSampler,
    BatchSampler,
    WeightedRandomSampler,
    # DistributedSampler,
    Sampler,
)
from paddle.io import DistributedBatchSampler
import paddle
import numpy as np
from typing import List
from speechbrain.dataio.dataset import DynamicItemDataset
from scipy.stats import lognorm

logger = logging.getLogger(__name__)

# in paddle, we must need BatchSampler
class ReproducibleRandomSampler(BatchSampler):
    """A modification of RandomSampler which always returns the same values.

    Also look at `torch.utils.data.RandomSampler`. This has mostly
    the same behaviour and arguments, except for adding 'seed' and 'epoch' and
    not supporting 'generator'.

    Note
    ----
    Call `set_epoch` before every epoch. Otherwise, the sampler will produce the
    same sequence of indices every epoch.

    Arguments
    ---------
    data_source : Dataset
        The data source to sample indices for.
    seed : int
        The base seed to use for the random number generator. It is recommended
        to use a value which has a good mix of 0 and 1 bits.
    epoch : int
        The epoch to start at.

    Example
    -------
    >>> import paddle
    >>> from speechbrain.utils.checkpoints import Checkpointer
    >>> from speechbrain.dataio.dataloader import SaveableDataLoader
    >>> # An example "dataset"
    >>> dataset = torch.arange(10).unsqueeze(1)
    >>> # Create the random sampler:
    >>> sampler = ReproducibleRandomSampler(dataset)
    >>> dataloader = SaveableDataLoader(dataset, sampler = sampler,
    ...     num_workers = 3)
    >>> # Setup the checkpointer.
    >>> # Note that the sampler doesn't need to be saved itself.
    >>> tmpdir = getfixture('tmpdir')
    >>> checkpointer = Checkpointer(tmpdir, {"dataloader": dataloader})
    >>> # Iterate:
    >>> subset = []
    >>> for i, data_point in enumerate(dataloader):
    ...     # Say you save a checkpoint on the fourth batch:
    ...     if i == 3:
    ...         _ = checkpointer.save_checkpoint(end_of_epoch = False)
    ...     # So let's save the numbers you would get if you continue
    ...     if i >= 4:
    ...         subset.append(data_point.item())
    >>> # What if instead you had to restart the experiment?
    >>> new_sampler = ReproducibleRandomSampler(dataset)
    >>> new_dataloader = SaveableDataLoader(dataset, sampler = new_sampler,
    ...        num_workers = 3)
    >>> new_checkpointer = Checkpointer(tmpdir, {"dataloader": new_dataloader})
    >>> _ = new_checkpointer.recover_if_possible()
    >>> # You'll get the same random order again:
    >>> new_subset = [data_point.item() for data_point in new_dataloader]
    >>> assert subset == new_subset

    """

    def __init__(self, data_source, seed=563375142, epoch=0, **kwargs):
        if "generator" in kwargs:
            MSG = (
                "Cannot give a separate generator when using "
                + "ReproducibleRandomSampler"
            )
            raise ValueError(MSG)
        # sampler = RandomSampler(data_source, **kwargs)
        super().__init__(data_source, **kwargs)
        self.seed = int(seed)
        self.epoch = epoch
        # self.generator = paddle.fluid.generator.Generator()

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self.epoch = epoch

    def __iter__(self):
        # batch_idx = super().__iter__()
        # self.generator.manual_seed(self.seed + self.epoch)
        return super().__iter__()


class ReproducibleWeightedRandomSampler(WeightedRandomSampler):
    """A reproducible modification of WeightedRandomSampler.

    Also look at `torch.utils.data.WeightedRandomSampler`. This has the
    the same behaviour and arguments, except for adding 'seed' and 'epoch' and
    not supporting 'generator'.

    Note
    ----
    Call `set_epoch` before every epoch. Otherwise, the sampler will produce the
    same sequence of indices every epoch.

    Arguments
    ---------
    weights : sequence of float
        Weights for each index. Doesn't need to sum to one.
    num_samples : int
        Number of samples to draw
    replacement : bool
        To draw with replacement or not (within an epoch of num_samples).
    seed : int
        The base seed to use for the random number generator. It is recommended
        to use a value which has a good mix of 0 and 1 bits.
    epoch : int
        The epoch to start at.

    Example
    -------
    >>> a = ReproducibleWeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True)
    >>> b = ReproducibleWeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True)
    >>> list(a)
    [3, 1, 4, 4, 4]
    >>> list(b)
    [3, 1, 4, 4, 4]
    >>> a.set_epoch(1)
    >>> list(a)
    [4, 5, 4, 4, 3]
    >>> b.set_epoch(1)
    >>> list(b)
    [4, 5, 4, 4, 3]


    """

    def __init__(
        self,
        weights,
        num_samples,
        replacement,
        seed=129491412,
        epoch=0,
        **kwargs,
    ):
        if "generator" in kwargs:
            MSG = (
                "Cannot give a separate generator when using "
                + "ReproducibleRandomSampler"
            )
            raise ValueError(MSG)
        super().__init__(weights, num_samples, replacement, **kwargs)
        self.seed = int(seed)
        self.epoch = epoch
        # self.generator = torch.Generator()

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self.epoch = epoch

    def __iter__(self):
        # self.generator.manual_seed(self.seed + self.epoch)
        return super().__iter__()


class ConcatDatasetBatchSampler(Sampler):
    """This sampler is built to work with a standard Pytorch ConcatDataset.

    It is used to retrieve elements from the different concatenated datasets placing them in the same batch
    with proportion specified by batch_sizes, e.g 8, 16 means each batch will
    be of 24 elements with the first 8 belonging to the first dataset in ConcatDataset
    object and the last 16 to the second.
    More than two datasets are supported, in that case you need to provide 3 batch
    sizes.

    Note
    ----
    Batched are drawn from the datasets till the one with smallest length is exhausted.
    Thus number of examples in your training epoch is dictated by the dataset
    whose length is the smallest.


    Arguments
    ---------
    samplers : int
        The base seed to use for the random number generator. It is recommended
        to use a value which has a good mix of 0 and 1 bits.
    batch_sizes: list
        Batch sizes.
    epoch : int
        The epoch to start at.

    Example
    -------
    >>> import paddle
    >>> from speechbrain.dataio.sampler import ConcatDatasetBatchSampler, ReproducibleRandomSampler
    >>> from speechbrain.dataio.sampler import ReproducibleRandomSampler
    >>> from speechbrain.dataio.dataloader import SaveableDataLoader
    >>> # example "datasets"
    >>> dataset1 = torch.arange(0, 10).unsqueeze(1)
    >>> dataset2 = torch.arange(20, 40).unsqueeze(1)
    >>> tot_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
    >>> sampler1 = ReproducibleRandomSampler(dataset1)
    >>> sampler2 = ReproducibleRandomSampler(dataset2)
    >>> tot_sampler = ConcatDatasetBatchSampler([sampler1, sampler2], [2, 4])
    >>> dataloader = SaveableDataLoader(tot_dataset, batch_sampler = tot_sampler,
    ...     num_workers = 3)
    >>> for data_point in dataloader:
    ...      assert len(data_point) == 6
    ...      for i in range(2):
    ...         assert data_point[i] in [x for x in range(0, 10)]
    ...      for i in range(2, 4):
    ...         assert data_point[i] in [x for x in range(10, 40)]
    """

    def __init__(self, samplers, batch_sizes: (tuple, list), epoch=0) -> None:

        if not isinstance(samplers, (list, tuple)):
            raise ValueError(
                "samplers should be a list or tuple of Pytorch Samplers, "
                "but got samplers={}".format(batch_sizes)
            )

        if not isinstance(batch_sizes, (list, tuple)):
            raise ValueError(
                "batch_sizes should be a list or tuple of integers, "
                "but got batch_sizes={}".format(batch_sizes)
            )

        if not len(batch_sizes) == len(samplers):
            raise ValueError(
                "batch_sizes and samplers should be have same length"
            )

        self.batch_sizes = batch_sizes
        self.samplers = samplers
        self.offsets = [0] + np.cumsum(
            [len(x) for x in self.samplers]
        ).tolist()[:-1]
        self.epoch = epoch
        self.set_epoch(self.epoch)

    def _iter_one_dataset(self, c_batch_size, c_sampler, c_offset):
        batch = []
        for idx in c_sampler:
            batch.append(c_offset + idx)
            if len(batch) == c_batch_size:
                yield batch

    def set_epoch(self, epoch):
        """You can also just access self.epoch, but we maintain this interface
        to mirror ``torch.utils.data.distributed.DistributedSampler``.
        """
        if hasattr(self.samplers[0], "epoch"):
            for s in self.samplers:
                s.set_epoch(epoch)

    def __iter__(self):

        iterators = [iter(i) for i in self.samplers]
        tot_batch = []
        for b_num in range(len(self)):
            for samp_idx in range(len(self.samplers)):
                c_batch = []
                while len(c_batch) < self.batch_sizes[samp_idx]:
                    # 在paddle 中， next返回的是一个列表
                    c_batch.extend(
                        [self.offsets[samp_idx] + x for x in next(iterators[samp_idx])]
                    )
                tot_batch.extend(c_batch)
            yield tot_batch
            tot_batch = []

    def __len__(self):

        min_len = float("inf")
        for idx, sampler in enumerate(self.samplers):
            c_len = len(sampler) // self.batch_sizes[idx]
            min_len = min(c_len, min_len)

        return min_len


class DynamicBatchSampler(Sampler):
    """This BatchSampler batches examples together by grouping them by their length.

    Every example in the batch have approximately the same length and
    thus padding is minimized.
    This enables faster training on datasets
    where length of examples can vary significantly (e.g Librispeech).
    Inspired by: https://www.tensorflow.org/api_docs/python/tf/data/experimental/bucket_by_sequence_length

    Dynamic batching is performed by specifying a max_batch_length which is the
    upper limit for the sum of the length of examples in a batch:
    e.g., if ex1 has length 4, ex2 length 5 and if max_batch_length is set to 6
    ex1 and ex2 will be placed, alone, in two distinct batches.

    Length for each example can be obtained in two manners.
    If the input dataset is a DynamicItemDataset it can be obtained by specifying a
    length_func. Default assumes a "duration" entry is in the annotation.
    Length for each example can also be passed to this class upon instantiation
    by specifying a list containing the length for each example and passing it to
    lengths_list.

    Examples are grouped together by defining a set of possible discrete intervals
    (buckets). Examples whose length fall into these intervals can be batched together.

    The number of buckets can be specified by using the arg num_buckets.
    There is usually an optimal range for the value of this argument.

    If num_buckets == 1, all examples can be batched together. You have maximum randomization
    but your training speed will be slower due to the fact that a large amount of the values will be padding
    as long and short examples can be batched together.
    As the number of buckets grows only examples with similar
    length can be grouped together.
    This trades-off speed with randomization.
    TLDR: Low number -> better randomization, High number -> faster training.
    NOTE THAT: if set too high the training speed will decrease. If num_buckets -> number of examples in the dataset the batch size
    will be small impacting training speed and possibly performance.

    The buckets can also be specified by passing a list to the bucket_boundaries
    argument instead of specifying a left_bucket_length and a bucket_length_multiplier.

    Example
    -------
    >>> import paddle
    >>> import speechbrain as sb
    >>> from speechbrain.dataio.sampler import DynamicBatchSampler
    >>> from speechbrain.dataio.dataset import DynamicItemDataset
    >>> from speechbrain.dataio.dataloader import SaveableDataLoader
    >>> from speechbrain.dataio.batch import PaddedBatch
    >>> import numpy as np
    >>> item_lengths = sorted([np.random.randint(10, 100) for x in range(20)])
    >>> dataset = {"ex_{}".format(x) : {"wav" :torch.randn(x)} for x in item_lengths}
    >>> dataset = DynamicItemDataset(dataset)
    >>> dataset.set_output_keys(["wav"])
    >>> length_func = lambda x : len(x) # trivial in this example
    >>> bsampler = DynamicBatchSampler(dataset, 20, 4, length_func, shuffle=False, batch_ordering='descending')
    >>> dataloader = SaveableDataLoader(dataset, batch_sampler=bsampler, collate_fn=PaddedBatch)
    >>> for i, b in enumerate(dataloader):
    ...     data, length = b["wav"]
    >>> assert data.shape[-1] == max(item_lengths)

    Arguments
    ---------
    dataset : torch.utils.data.Dataset
        Pytorch Dataset from which elements will be sampled.
    max_batch_length : int
        Upper limit for the sum of the length of examples in a batch.
        Should be chosen based on your GPU memory.
    num_buckets : int
        Number of discrete buckets used to group examples together.
        If num_buckets == 1, all examples can be batched together. As the number of buckets grows only examples with similar
        length can be grouped together. This trades-off speed with randomization.
        Low number -> better randomization, High number -> faster training.
        However if set too high the training speed will decrease. If num_buckets -> number of examples in the dataset the batch size
        will be small impacting training speed and possibly performance.
        NOTE: you have either to specify manually the bucket_boundaries or the number of buckets.
    length_func : callable
        Function used to get length of each example from the dataset.
        This argument can be used only when the dataset is a Speechbrain DynamicItemDataset object.
        Can be anything: e.g. lambda x: x["duration"]*16000 returns number of samples
        if duration key in the annotation is in seconds and the file has 16kHz sampling freq.
    shuffle : bool
        Whether or not shuffle examples between each epoch.
    batch_ordering : string
        If ``random``, batches are randomly permuted; otherwise ``ascending`` or ``descending`` sorted by length.
    max_batch_ex: int
        If set, it limits the maximum number of examples that can be in a batch superseeding max_batch_length
        in instances where the amount of examples will exceeed the value specified here.
        E.g. you have a lot of short examples and the batch size for those will be too high, you can use this argument
        to limit the batch size for these short examples.
    bucket_boundaries : list
        Overrides bucket_length_multiplier and left_bucket_length by specifying manually
        the buckets right boundaries.
    lengths_list: list
        Overrides length_func by passing a list containing the length of each example
        in the dataset. This argument must be set when the dataset is a plain
        Pytorch Dataset object and not a DynamicItemDataset object as length_func
        cannot be used on Pytorch Datasets.
    epoch : int
        The epoch to start at.
    drop_last : bool
         If ``True``, the sampler will drop the last examples which
         have not been grouped.
    verbose: bool
        If ``True``, log also the stats for each batch at the first epoch.
    """

    def __init__(
        self,
        dataset,
        max_batch_length: int,
        num_buckets: int = None,
        length_func=lambda x: x["duration"],
        shuffle: bool = True,
        batch_ordering: str = "random",
        max_batch_ex: int = -1,
        bucket_boundaries: List[int] = [],
        lengths_list: List[int] = None,
        seed: int = 42,
        epoch: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
    ):
        self._dataset = dataset
        self._ex_lengths = {}
        ex_ids = self._dataset.data_ids
        self.verbose = verbose

        # We do not put a default on num_buckets to encourage users to play with this parameter
        if num_buckets is None and len(bucket_boundaries) == 0:
            raise RuntimeError(
                "Please specify either num_buckets or bucket boundaries."
                "Check the docs, and/or the tutorial !"
            )

        if lengths_list is not None:
            # take length of examples from this argument and bypass length_key
            for indx in range(len(lengths_list)):
                self._ex_lengths[str(indx)] = lengths_list[indx]
        else:
            # use length func
            if not isinstance(dataset, DynamicItemDataset):
                raise NotImplementedError(
                    "Dataset should be a Speechbrain DynamicItemDataset when using length function"
                )
            for indx in range(len(self._dataset)):
                self._ex_lengths[str(indx)] = length_func(
                    self._dataset.data[ex_ids[indx]]
                )

        if len(bucket_boundaries) > 0:
            if not all([x >= 0 for x in bucket_boundaries]):
                raise ValueError(
                    "All elements in bucket boundaries should be non-negative (>= 0)."
                )
            if not len(set(bucket_boundaries)) == len(bucket_boundaries):
                raise ValueError(
                    "Bucket_boundaries should not contain duplicates."
                )
            np.testing.assert_array_equal(
                np.array(bucket_boundaries),
                np.array(sorted(bucket_boundaries)),
                err_msg="The arg bucket_boundaries should be an ascending sorted list of non negative values values!",
            )
            self._bucket_boundaries = np.array(sorted(bucket_boundaries))
        else:
            # use num_buckets
            self._bucket_boundaries = np.array(
                self._get_boundaries_through_warping(
                    max_batch_length=max_batch_length,
                    num_quantiles=num_buckets,
                )
            )

        self._max_batch_length = max_batch_length
        self._shuffle_ex = shuffle
        self._batch_ordering = batch_ordering
        self._seed = seed
        self._drop_last = drop_last
        self._max_batch_ex = max_batch_ex
        # Calculate bucket lengths - how often does one bucket boundary fit into max_batch_length?
        self._bucket_lens = [
            max(1, int(max_batch_length / self._bucket_boundaries[i]))
            for i in range(len(self._bucket_boundaries))
        ] + [1]
        self._epoch = epoch
        self._generate_batches()

    def get_durations(self, batch):
        return [self._ex_lengths[str(idx)] for idx in batch]

    def _get_boundaries_through_warping(
        self, max_batch_length: int, num_quantiles: int,
    ) -> List[int]:

        # NOTE: the following lines do not cover that there is only one example in the dataset
        # warp frames (duration) distribution of train data
        logger.info("Batch quantisation in latent space")
        # linspace set-up
        num_boundaries = num_quantiles + 1
        # create latent linearly equal spaced buckets
        latent_boundaries = np.linspace(
            1 / num_boundaries, num_quantiles / num_boundaries, num_quantiles,
        )
        # get quantiles using lognormal distribution
        quantiles = lognorm.ppf(latent_boundaries, 1)
        # scale up to to max_batch_length
        bucket_boundaries = quantiles * max_batch_length / quantiles[-1]
        # compute resulting bucket length multipliers
        length_multipliers = [
            bucket_boundaries[x + 1] / bucket_boundaries[x]
            for x in range(num_quantiles - 1)
        ]
        # logging
        logger.info(
            "Latent bucket boundary - buckets: {} - length multipliers: {}".format(
                list(map("{:.2f}".format, bucket_boundaries)),
                list(map("{:.2f}".format, length_multipliers)),
            )
        )
        return list(sorted(bucket_boundaries))

    def _permute_batches(self):

        if self._batch_ordering == "random":
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(
                len(self._batches), generator=g
            ).tolist()  # type: ignore
            tmp = []
            for idx in sampler:
                tmp.append(self._batches[idx])
            self._batches = tmp

        elif self._batch_ordering == "ascending":
            self._batches = sorted(
                self._batches,
                key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
            )
        elif self._batch_ordering == "descending":
            self._batches = sorted(
                self._batches,
                key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
                reverse=True,
            )
        else:
            raise NotImplementedError

    def _generate_batches(self):
        logger.info("DynamicBatchSampler: Generating dynamic batches")
        if self._shuffle_ex:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(len(self._dataset), generator=g).tolist()  # type: ignore
        else:
            # take examples as they are: e.g. they have been sorted
            sampler = range(len(self._dataset))  # type: ignore

        self._batches = []
        bucket_batches = [[] for i in self._bucket_lens]

        stats_tracker = [
            {"min": np.inf, "max": -np.inf, "tot": 0, "n_ex": 0}
            for i in self._bucket_lens
        ]

        for idx in sampler:
            # length of pre-sampled audio
            item_len = self._ex_lengths[str(idx)]
            # bucket to fill up most padding
            bucket_id = np.searchsorted(self._bucket_boundaries, item_len)
            # fill audio's duration into that bucket
            bucket_batches[bucket_id].append(idx)

            stats_tracker[bucket_id]["min"] = min(
                stats_tracker[bucket_id]["min"], item_len
            )
            stats_tracker[bucket_id]["max"] = max(
                stats_tracker[bucket_id]["max"], item_len
            )
            stats_tracker[bucket_id]["tot"] += item_len
            stats_tracker[bucket_id]["n_ex"] += 1
            # track #samples - why not duration/#frames; rounded up?
            # keep track of durations, if necessary

            if (
                len(bucket_batches[bucket_id]) >= self._bucket_lens[bucket_id]
                or len(bucket_batches[bucket_id]) <= self._max_batch_ex
            ):
                self._batches.append(bucket_batches[bucket_id])
                bucket_batches[bucket_id] = []
                # keep track of durations

        # Dump remaining batches
        if not self._drop_last:
            for batch in bucket_batches:
                if batch:
                    self._batches.append(batch)

        self._permute_batches()  # possibly reorder batches

        if self._epoch == 0:  # only log at first epoch
            # frames per batch & their padding remaining
            boundaries = [0] + self._bucket_boundaries.tolist()

            for bucket_indx in range(len(self._bucket_boundaries)):
                try:
                    num_batches = stats_tracker[bucket_indx]["tot"] // (
                        self._max_batch_length
                    )
                    pad_factor = (
                        stats_tracker[bucket_indx]["max"]
                        - stats_tracker[bucket_indx]["min"]
                    ) / (
                        stats_tracker[bucket_indx]["tot"]
                        / stats_tracker[bucket_indx]["n_ex"]
                    )
                except ZeroDivisionError:
                    num_batches = 0
                    pad_factor = 0

                logger.info(
                    (
                        "DynamicBatchSampler: Bucket {} with boundary {:.1f}-{:.1f} and "
                        + "batch_size {}: Num Examples {:.1f}, Num Full Batches {:.3f}, Pad Factor {:.3f}."
                    ).format(
                        bucket_indx,
                        boundaries[bucket_indx],
                        boundaries[bucket_indx + 1],
                        self._bucket_lens[bucket_indx],
                        stats_tracker[bucket_indx]["n_ex"],
                        num_batches,
                        pad_factor * 100,
                    )
                )

            if self.verbose:
                batch_stats = {
                    "tot_frames": [],
                    "tot_pad_frames": [],
                    "pad_%": [],
                }
                for batch in self._batches:
                    tot_frames = sum(
                        [self._ex_lengths[str(idx)] for idx in batch]
                    )
                    batch_stats["tot_frames"].append(tot_frames)
                    max_frames = max(
                        [self._ex_lengths[str(idx)] for idx in batch]
                    )
                    tot_pad = sum(
                        [
                            max_frames - self._ex_lengths[str(idx)]
                            for idx in batch
                        ]
                    )
                    batch_stats["tot_pad_frames"].append(tot_pad)
                    batch_stats["pad_%"].append(tot_pad / tot_frames * 100)

                padding_details = "Batch {} with {:.1f} frames with {} files - {:.1f} padding, {:.2f} (%) of total."
                padding_details = "DynamicBatchSampler: " + padding_details
                for i in range(len(self._batches)):
                    logger.info(
                        padding_details.format(
                            i,
                            batch_stats["tot_frames"][i],
                            len(self._batches[i]),
                            batch_stats["tot_pad_frames"][i],
                            batch_stats["pad_%"][i],
                        )
                    )

    def __iter__(self):
        for batch in self._batches:
            yield batch
        if self._shuffle_ex:  # re-generate examples if ex_ordering == "random"
            self._generate_batches()
        if self._batch_ordering == "random":
            # we randomly permute the batches only --> faster
            self._permute_batches()
        else:
            pass

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self._epoch = epoch
        self._generate_batches()

    def __len__(self):
        return len(self._batches)


class DistributedSampler(DistributedBatchSampler):
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 drop_last=False):
        super().__init__(
            dataset=dataset,
            batch_size=1,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last)

# # Heavily inspired by Catalyst, which is under Apache 2.0 licence.
# # https://github.com/catalyst-team/catalyst/blob/51428d7756e62b9b8ee5379f38e9fd576eeb36e5/catalyst/data/sampler.py#L522
class DistributedSamplerWrapper(DistributedSampler):
    """This wrapper allows using any sampler with Distributed Data Parallel (DDP) correctly.

    Passing blindly the sampler to each DDP process will cause to have access
    within each process to all the data in the dataset instead of only a subset
    of it which is unique to each process.  This wrapper prevents this and
    allows to use only a subset of the original data for each process.

    NOTE
    ----
    This is is automatically applied to any sampler in the Brain class when DDP
    training is used.
    """

    def __init__(self, sampler, *args, **kwargs):
        # DistributedSampler only calls len() on dataset
        # so a sampler is fine to pass there, as well.
        super().__init__(dataset=sampler, *args, **kwargs)
        self.sampler = sampler

    def __iter__(self):
        # It is easiest to use a random access interface to the wrapped
        # sampler's indices, so we just fetch all indices from the wrapped
        # sampler
        sampler_indices = list(self.sampler.__iter__())
        indices_of_indices = super().__iter__()
        # Itemgetter fetches the wrapped sampler indices from the positions
        # pointed to by DistributedSampler
        return iter(itemgetter(*indices_of_indices)(sampler_indices))

    def set_epoch(self, epoch):
        """Pass set_epoch() through to DistributedSampler and the wrapper one"""
        super().set_epoch(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
