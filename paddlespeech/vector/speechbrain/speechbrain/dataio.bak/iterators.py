"""Webdataset compatible iterators

Authors:
 * Aku Rouhe 2021
"""
import bisect
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Any
from speechbrain.dataio.batch import PaddedBatch


@dataclass(order=True)
class LengthItem:
    length: int
    data: Any = field(compare=False)


def total_length_with_padding(lengths):
    # How long would batch be (with padding)
    return len(lengths) * max(lengths)


def padding_ratio(lengths):
    # How much of batch is padding:
    return 1.0 - sum(lengths) / total_length_with_padding(lengths)


@dataclass(order=True)
class RatioIndex:
    ratio: float
    index: int


def indices_around_random_pivot(
    databuffer,
    target_batch_numel,
    max_batch_size=None,
    max_batch_numel=None,
    max_padding_ratio=0.2,
    randint_generator=random.randint,
):
    """Random pivot sampler_fn for dynamic_bucketed_batch

    Create a batch around a random pivot index in the sorted buffer

    This works on the databuffer which is assumed to be in sorted order. An
    index is chosen at random. This starts the window of indices: at first,
    only the randomly chosen pivot index is included. The window of indices is
    grown one-index-at-a-time, picking either the index to the right of the
    window, or the index to the left, picking the index that would increase the
    padding ratio the least, and making sure the batch wouldn't exceed the
    maximum batch length nor the maximum padding ratio.

    Arguments
    ---------
    databuffer : list
        Sorted list of LengthItems
    target_batch_numel : int
        Target of total batch length including padding, which is simply computed
        as batch size * length of longest example. This function aims to return
        the batch as soon as the gathered length exceeds this. If some limits
        are encountered first, this may not be satisifed.
    max_batch_size : None, int
        Maximum number of examples to include in the batch, or None to not limit
        by number of examples.
    max_batch_numel : None, int
        Maximum of total batch length including padding, which is simply computed
        as batch size * length of longest example.

    """
    bufferlen = len(databuffer)
    if max_batch_size is None:
        max_batch_size = bufferlen
    # Choose pivot:
    min_index = max_index = randint_generator(0, bufferlen - 1)
    lengths = [databuffer[min_index].length]

    # Define index filtering function:
    def possibly_consider(index, to_consider):
        # Adds an index to the to_consider list,
        # if the index passes all requirements.
        if index < 0 or index >= len(databuffer):
            return
        consideree = databuffer[index]
        updated_lengths = [consideree.length] + lengths
        if max_batch_numel is not None:
            updated_total = total_length_with_padding(updated_lengths)
            if updated_total > max_batch_numel:
                return
        updated_ratio = padding_ratio(updated_lengths)
        if max_padding_ratio is not None and updated_ratio > max_padding_ratio:
            return
        to_consider.append(RatioIndex(updated_ratio, index))

    # Loop till the target length is exceeded or max batch size is hit:
    while (
        max_index + 1 - min_index < max_batch_size
        and total_length_with_padding(lengths) < target_batch_numel
    ):
        # Consider indices to the left and to the right, if they
        # pass the requirements:
        to_consider = []
        possibly_consider(min_index - 1, to_consider)
        possibly_consider(max_index + 1, to_consider)
        # If neither pass the requirements, then we must return the batch
        # as it is now (there can be no better addition):
        if not to_consider:
            break
        # Pick the index that minimizes the padding ratio increase:
        to_add = min(to_consider)
        min_index = min(min_index, to_add.index)
        max_index = max(max_index, to_add.index)
        lengths.append(databuffer[to_add.index].length)
    return list(range(min_index, max_index + 1))


def dynamic_bucketed_batch(
    data,
    len_key=None,
    len_fn=len,
    min_sample_len=None,
    max_sample_len=None,
    buffersize=1024,
    collate_fn=PaddedBatch,
    sampler_fn=indices_around_random_pivot,
    sampler_kwargs={},
    drop_end=False,
):
    """Produce batches from a sorted buffer

    This function keeps a sorted buffer of the incoming samples.
    The samples can be filtered for min/max length.
    An external sampler is used to choose samples for each batch,
    which allows different dynamic batching algorithms to be used.

    Arguments
    ---------
    data : iterable
        An iterable source of samples, such as an IterableDataset.
    len_key : str, None
        The key in the sample dict to use to fetch the length of the sample, or
        None if no key should be used.
    len_fn : callable
        Called with sample[len_key] if len_key is not None, else sample. Needs
        to return the sample length as an integer.
    min_sample_len : int, None
        Discard samples with length lower than this. If None, no minimum is
        applied.
    max_sample_len : int, None
        Discard samples with length larger than this. If None, no maximum is
        applied.
    buffersize : int
        The size of the internal sorted buffer. The buffer is always filled up
        before yielding a batch of samples.
    collate_fn : callable
        Called with a list of samples. This should return a batch. By default, using
        the SpeechBrain PaddedBatch class, which works for dict-like samples, and
        pads any tensors.
    sampler_fn : callable
        Called with the sorted data buffer. Needs to return a list of indices, which
        make up the next batch. By default using ``indices_around_random_pivot``
    sampler_kwargs : dict
        Keyword arguments, passed to sampler_fn.
    drop_end : bool
        After the data stream is exhausted, should batches be made until the data
        buffer is exhausted, or should the rest of the buffer be discarded. Without
        new samples, the last batches might not be efficient to process.
        Note: you can use ``.repeat`` on `webdataset` IterableDatasets to never
        run out of new samples, and then use
        `speechbrain.dataio.dataloader.LoopedLoader` to set a nominal epoch length.
    """
    databuffer = []
    if sampler_kwargs:
        sampler_fn = partial(sampler_fn, **sampler_kwargs)
    for sample in data:
        # Length fetching interface has multiple valid call signatures:
        if len_key is not None and len_fn is not None:
            length = len_fn(sample[len_key])
        elif len_key is not None:
            length = sample[len_key]
        elif len_fn is not None:
            length = len_fn(sample)
        else:
            raise ValueError("Must specify at least one of len_key or len_fn")
        # Possibly filter by length:
        if (min_sample_len is not None and length < min_sample_len) or (
            max_sample_len is not None and length > max_sample_len
        ):
            # Drop sample
            continue
        item = LengthItem(length, sample)
        # bisect.insort inserts in sorted order.
        # This should be a good way to maintain a sorted list,
        # but perhaps simply filling up the buffer and calling .sort()
        # could be good as well (Python's sort leverages already sorted segments)
        bisect.insort(databuffer, item)
        if len(databuffer) == buffersize:
            indices = sampler_fn(databuffer)
            batch_list = []
            # popping from highest to lowest is safe
            for i in sorted(indices, reverse=True):
                item = databuffer.pop(i)
                batch_list.append(item.data)
            yield collate_fn(batch_list)
    # Data stream was exhausted. Data buffer is relatively full at first,
    # but cannot be replenished, so batches might not be efficiently produced.
    # Either stop, or exhaust buffer.
    if drop_end:
        return
    while databuffer:
        indices = sampler_fn(databuffer)
        batch_list = []
        for i in sorted(indices, reverse=True):
            item = databuffer.pop(i)
            batch_list.append(item.data)
        yield collate_fn(batch_list)
