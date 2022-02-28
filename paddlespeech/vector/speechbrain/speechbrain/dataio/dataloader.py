"""PyTorch compatible DataLoaders

Essentially we extend PyTorch DataLoader by adding the ability to save the
data loading state, so that a checkpoint may be saved in the middle of an
epoch.

Example
-------
>>> import paddle
>>> from speechbrain.utils.checkpoints import Checkpointer
>>> # An example "dataset" and its loader
>>> dataset = torch.randn(10, 1)
>>> dataloader = SaveableDataLoader(dataset, num_workers = 3)
>>> # Setup the checkpointer:
>>> tmpdir = getfixture('tmpdir')
>>> checkpointer = Checkpointer(tmpdir, {"dataloader": dataloader})
>>> # Iterate:
>>> for i, data_point in enumerate(dataloader):
...     # Here you would process the data:
...     rainfall_amount_prediction = data_point * 4.
...     # Now, imagine the experiment gets killed on the fifth batch:
...     if i == 4:
...         break
...     # Luckily, you had just saved a checkpoint:
...     if i == 3:
...         _ = checkpointer.save_checkpoint(end_of_epoch = False)
>>> # So when you restart the experiment:
>>> new_dataloader = SaveableDataLoader(dataset, num_workers = 3)
>>> new_checkpointer = Checkpointer(tmpdir, {"dataloader": new_dataloader})
>>> _ = new_checkpointer.recover_if_possible()
>>> # The dataloader fast-forwards to the position where we left off:
>>> assert next(iter(new_dataloader)) == dataset[4]

Authors:
  * Aku Rouhe 2020
"""
import paddle
from paddle.io import DataLoader
from paddle.io import IterableDataset
from speechbrain.dataio.dataset import SamplesDataset
from paddle.fluid.dataloader.dataloader_iter import _DataLoaderIterBase
from paddle.fluid.dataloader.dataloader_iter import _DataLoaderIterSingleProcess
from paddle.fluid.dataloader.dataloader_iter import _DataLoaderIterMultiProcess
import logging
import warnings
import functools
import sys
from speechbrain.dataio.batch import PaddedBatch, BatchsizeGuesser, padded_batch_collate
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.sampler import ReproducibleRandomSampler
from speechbrain.utils.checkpoints import (
    register_checkpoint_hooks,
    mark_as_saver,
    mark_as_loader,
)

# Optional support for webdataset
try:
    import webdataset as wds

    WDS_AVAILABLE = True
except ImportError:
    WDS_AVAILABLE = False

logger = logging.getLogger(__name__)


def make_dataloader(dataset, looped_nominal_epoch=None, **loader_kwargs):
    """Makes a basic DataLoader with SpeechBrain defaults.

    For DynamicItemDatasets (which return dicts), use
    PaddedBatch as the default collate_fn.

    Shuffling gets implemented by ReproducibleRandomSampler.

    If the Dataset is not an IterableDataset, the DataLoader
    is a SaveableDataLoader.

    If the Dataset is a webdataset.dataset.Composable, set default
    batch_size = None.

    Can also loop over the underlying dataloader continuously,
    and stop iterations at nominal epoch lengths.

    Arguments
    ---------
    dataset : Dataset
        The dataset to make a DataLoader for.
    looped_nominal_epoch : None, int
        If an integer is given, loop the underlying DataLoader infinitely and
        set a nominal epoch length in batches (or whatever the DataLoader
        yields).
    **loader_kwargs : dict
        Keyword args to DataLoader, see PyTorch DataLoader for
        options.

    Returns
    -------
    DataLoader
        If looped_nominal_epoch is None
    LoopedLoader
        If looped_nominal_epoch is not None
    """
    # PaddedBatch as default collation for DynamicItemDataset
    if "collate_fn" not in loader_kwargs and isinstance(
        dataset, DynamicItemDataset
    ):
        # loader_kwargs["collate_fn"] = PaddedBatch
        # 这里Paddle可能是有bug，还是有其他更合适的调用方式？？？
        loader_kwargs["collate_fn"] = padded_batch_collate
    # Reproducible random sampling
    # logger.info("loader kwargs: {}".format(loader_kwargs))
    sampler_kwargs = {}
    # if loader_kwargs.get("shuffle", False):
    #     if loader_kwargs.get("sampler") is not None:
    #         raise ValueError(
    #             "Cannot specify both shuffle=True and a "
    #             "sampler in loader_kwargs"
            # )
    # logger.info("loader kwargs: {}".format(loader_kwargs))   
    if loader_kwargs.get("batch_size", None) != None:
        sampler_kwargs["batch_size"] = loader_kwargs["batch_size"]
        del loader_kwargs["batch_size"]
        
    if loader_kwargs.get("shuffle", None) != None:
        sampler_kwargs["shuffle"] = loader_kwargs["shuffle"]
        del loader_kwargs["shuffle"]            
        
    if loader_kwargs.get("drop_last", None) != None:
        sampler_kwargs["drop_last"] = loader_kwargs["drop_last"]
        del loader_kwargs["drop_last"]            

    sampler = ReproducibleRandomSampler(dataset, **sampler_kwargs)
        # 在paddle中没哟sampler这个参数，只有batch_sampler这个参数
        # loader_kwargs["sampler"] = sampler
    loader_kwargs["batch_sampler"] = sampler
        # Should delete shuffle because you can't set both Sampler and
        # shuffle
        # NOTE: the dict of loader options may get used elsewhere!
        # However, this del doesn't touch those because loader_kwargs comes
        # from a **kwargs dict.
        # del loader_kwargs["shuffle"]
    # With WDS it is recommended to do batching in the dataset itself,
    # which requires batch_size = None in the DataLoader
    if (
        WDS_AVAILABLE
        and isinstance(dataset, wds.dataset.Composable)
        and "batch_size" not in loader_kwargs
    ):
        loader_kwargs["batch_size"] = None
    # Create the loader
    if isinstance(dataset, IterableDataset):
        # logger.info("create the DataLoader")
        dataloader = DataLoader(dataset, **loader_kwargs)
    else:
        # print("SaveableDataLoader")
        # logger.info("create the SaveableDataLoader, {}".format(loader_kwargs))
        # print("loader kwargs: {}".format(loader_kwargs))
        if not isinstance(dataset, paddle.io.Dataset):
            dataset = SamplesDataset(dataset)
        # logger.info("loader kwargs: {}".format(loader_kwargs))
        dataloader = SaveableDataLoader(dataset, **loader_kwargs)
    if looped_nominal_epoch is not None:
        dataloader = LoopedLoader(dataloader, looped_nominal_epoch)
    return dataloader


# We essentially want to make the DataLoader iterators able to skip ahead
# after checkpoint recovery
# This should be handled by the DataLoader iterators' base class.
# To make the implementation here a little more maintainable
# we decide to patch some PyTorch functionality


def __new_init(self, loader, *args, **kwargs):
    self.__old_init__(loader, *args, **kwargs)
    if (
        hasattr(loader, "_speechbrain_recovery_skip_to")
        and loader._speechbrain_recovery_skip_to is not None
    ):
        print("_speechbrain_recovery_skip_to: {}".format(loader._speechbrain_recovery_skip_to))
        # loader._speechbrain_recovery_skip_to = 1
        # Fast forward the sampler iterator since we have recovered:
        for i in range(loader._speechbrain_recovery_skip_to):
            try:
                next(self._sampler_iter)
            except StopIteration:
                MSG = "Tried to fast-forward Sampler after checkpoint "
                f"recovery by {loader._speechbrain_recovery_skip_to} "
                "indices, but now Sampler raised StopIteration after "
                f"{i} indices. Ignoring this mismatch."
                warnings.warn(MSG)
                break
            self._num_yielded = i + 1
        # Mark recovery as done:
        loader._speechbrain_recovery_skip_to = None


def __new_reset(self, loader, first_iter=False, *args, **kwargs):
    # On the first iteration, these have already normally been set by the init anyway.
    # And we don't want to overwrite them if we've recovered
    if not first_iter:
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

def __new_next(self):
    if not hasattr(self, "_num_yielded"):
        self._num_yielded = 1
    else:
        self._num_yielded += 1
    return self.__old_next__()
    

# functools.update_wrapper is meant for decorators, but it should basically
# preserve what we want:
functools.update_wrapper(__new_init, _DataLoaderIterBase.__init__)
_DataLoaderIterBase.__old_init__ = _DataLoaderIterBase.__init__
_DataLoaderIterBase.__init__ = __new_init
if hasattr(_DataLoaderIterBase, "_reset"):
    _DataLoaderIterBase._reset = __new_reset

# single process
functools.update_wrapper(__new_next, _DataLoaderIterSingleProcess.__next__)
_DataLoaderIterSingleProcess.__old_next__ = _DataLoaderIterSingleProcess.__next__
_DataLoaderIterSingleProcess.__next__ = __new_next

# multi-process
functools.update_wrapper(__new_next, _DataLoaderIterMultiProcess.__next__)
_DataLoaderIterMultiProcess.__old_next__ = _DataLoaderIterMultiProcess.__next__
_DataLoaderIterMultiProcess.__next__ = __new_next

@register_checkpoint_hooks
class SaveableDataLoader(DataLoader):
    """A saveable version of the PyTorch DataLoader.

    See `torch.utils.data.DataLoader` for usage. This class should work exactly
    like the PyTorch basic DataLoader, but this can be checkpointed with
    SpeechBrain's Checkpointer.

    Note
    ----
    1. The saveability is implemented via some unfortunately slightly magical
    means.
    2. The data loader cannot recover after entering __iter__. Normally this is
    not a problem, as recovery should happen before training begins.  However,
    just before evaluation, it is also typical to recover the checkpoint at
    which performance was the best. Thus, if a checkpoint is loaded after
    entering __iter__, we just assume it is for this reason. A warning is
    logged, but that is all.
    """

    def __init__(self, *args, **kwargs):
        # logger.info("Dataloader kwargs: {}".format(kwargs))
        super().__init__(*args, **kwargs)
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "SaveableDataLoader cannot save the position in an "
                "IterableDataset. Save the position on the dataset itself."
            )
        self._speechbrain_recovery_skip_to = None
        self._speechbrain_iterator = None

    def __iter__(self):
        iterator = super().__iter__()
        iterator._num_yielded = 0
        # Keep a reference to the iterator,
        # to be able to access the iterator._num_yielded value.
        # Keep a full reference (keeping the iterator alive)
        # rather than e.g. a weakref, as we may want to save a checkpoint
        # after the iterator has been exhausted, but before the full epoch has
        # ended (e.g. validation is still running)
        self._speechbrain_iterator = iterator
        # print("return the SaveableDataLoader iter instance")
        return iterator

    @mark_as_saver
    def _speechbrain_save(self, path):
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "Warning again: a checkpoint was requested on "
                "SaveableDataLoader, but the dataset is an IterableDataset. "
                "Cannot save the position in an IterableDataset. Not raising "
                "an error; assuming that you know what you're doing."
            )
        if self._speechbrain_iterator is None:
            to_save = None
        else:
            to_save = self._speechbrain_iterator._num_yielded
            # print("to save: {}".format(to_save))
        with open(path, "w") as fo:
            fo.write(str(to_save))

    @mark_as_loader
    def _speechbrain_load(self, path, end_of_epoch, device=None):
        del device  # Unused here
        if self._speechbrain_iterator is not None:
            logging.debug(
                "SaveableDataLoader was requested to load a "
                "checkpoint, but the DataLoader has already been "
                "iterated. The DataLoader file will be ignored. "
                "This is normal in evaluation, when a checkpoint is "
                "loaded just to retrieve the best model."
            )
            return
        if end_of_epoch:
            # Don't load at end of epoch, as we actually want to start a fresh
            # epoch iteration next.
            return
        with open(path) as fi:
            saved = fi.read()
            if saved == str(None):
                # Saved at a point where e.g. an iterator did not yet exist.
                return
            else:
                self._speechbrain_recovery_skip_to = int(saved)


@register_checkpoint_hooks
class LoopedLoader:
    """Loops an underlying iterable indefinitely, with nominal epoch lengths

    This is useful for working with IterableDatasets, and particularly
    webdataset-style loading. We recommend using ``.repeat()`` on the
    webdataset IterableDataset instance, so that the underlying dataloader
    naturally continues for ever.

    Arguments
    ---------
    loader : iterable
        A DataLoader or other iterable that is looped repeatedly.
    epoch_length : int
        The length of the nominal epoch. After this many steps, raises
        StopIteration
    """

    def __init__(self, loader, epoch_length, batchsize_fn=None):
        self.loader = loader
        self.iterator = None
        self.epoch_length = epoch_length
        self.step = 0  # Step in epoch
        self.total_steps = 0  # Total steps ever
        self.total_samples = 0  # Total samples seen on this process
        if batchsize_fn is None:
            self.batchsize_fn = BatchsizeGuesser()

    def __iter__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        return self

    def __next__(self):
        if self.step < self.epoch_length:
            self.step += 1
            self.total_steps += 1
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                batch = next(self.iterator)
            self.total_samples += self.batchsize_fn(batch)
            return batch
        else:
            self.step = 0
            raise StopIteration

    def __len__(self):
        return self.epoch_length

    @mark_as_saver
    def save(self, path):
        with open(path, "w") as fo:
            print(self.step, file=fo)
            print(self.total_steps, file=fo)
            print(self.total_samples, file=fo)

    @mark_as_loader
    def load(self, path, end_of_epoch=True, device=None):
        del device  # Unused here
        with open(path) as fi:
            self.step = int(fi.readline().strip())
            self.total_steps = int(fi.readline().strip())
            self.total_samples = int(fi.readline().strip())
            if not end_of_epoch and self.step == 0 and self.total_steps > 0:
                # Step has been set to 0 at the end of iteration,
                # so return it to epoch_length, so that first iteration
                # of this will immediately raise StopIteration.
                # Basically, this can happen when e.g. the main training
                # loop has already finished but there is a checkpoint in the
                # middle of validation.
                self.step = self.epoch_length
