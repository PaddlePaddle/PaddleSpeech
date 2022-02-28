"""Implements a checkpointable epoch counter (loop), optionally integrating early stopping.

Authors
 * Aku Rouhe 2020
 * Davide Borra 2021
"""
from .checkpoints import register_checkpoint_hooks
from .checkpoints import mark_as_saver
from .checkpoints import mark_as_loader
import logging

logger = logging.getLogger(__name__)


@register_checkpoint_hooks
class EpochCounter:
    """An epoch counter which can save and recall its state.

    Use this as the iterator for epochs.
    Note that this iterator gives you the numbers from [1 ... limit] not
    [0 ... limit-1] as range(limit) would.

    Example
    -------
    >>> from speechbrain.utils.checkpoints import Checkpointer
    >>> tmpdir = getfixture('tmpdir')
    >>> epoch_counter = EpochCounter(10)
    >>> recoverer = Checkpointer(tmpdir, {"epoch": epoch_counter})
    >>> recoverer.recover_if_possible()
    >>> # Now after recovery,
    >>> # the epoch starts from where it left off!
    >>> for epoch in epoch_counter:
    ...     # Run training...
    ...     ckpt = recoverer.save_checkpoint()
    """

    def __init__(self, limit):
        self.current = 0
        self.limit = int(limit)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.limit:
            self.current += 1
            logger.info(f"Going into epoch {self.current}")
            return self.current
        raise StopIteration

    @mark_as_saver
    def _save(self, path):
        with open(path, "w") as fo:
            fo.write(str(self.current))

    @mark_as_loader
    def _recover(self, path, end_of_epoch=True, device=None):
        # NOTE: end_of_epoch = True by default so that when
        #  loaded in parameter transfer, this starts a new epoch.
        #  However, parameter transfer to EpochCounter should
        #  probably never be used really.
        del device  # Not used.
        with open(path) as fi:
            saved_value = int(fi.read())
            if end_of_epoch:
                self.current = saved_value
            else:
                self.current = saved_value - 1


class EpochCounterWithStopper(EpochCounter):
    """An epoch counter which can save and recall its state, integrating an early stopper by tracking a target metric.

    Arguments
    ---------
    limit: int
        maximum number of epochs
    limit_to_stop : int
        maximum number of consecutive epochs without improvements in performance
    limit_warmup : int
        number of epochs to wait until start checking for early stopping
    direction : "max" or "min"
        direction to optimize the target metric

    Example
    -------
    >>> limit = 10
    >>> limit_to_stop = 5
    >>> limit_warmup = 2
    >>> direction = "min"
    >>> epoch_counter = EpochCounterWithStopper(limit, limit_to_stop, limit_warmup, direction)
    >>> for epoch in epoch_counter:
    ...     # Run training...
    ...     # Track a validation metric,
    ...     current_valid_metric = 0
    ...     # get the current valid metric (get current_valid_metric)
    ...     if epoch_counter.should_stop(current=epoch,
    ...                                  current_metric=current_valid_metric,):
    ...         epoch_counter.current = epoch_counter.limit  # skipping unpromising epochs
    """

    def __init__(self, limit, limit_to_stop, limit_warmup, direction):
        super().__init__(limit)
        self.limit_to_stop = limit_to_stop
        self.limit_warmup = limit_warmup
        self.direction = direction

        self.best_limit = 0
        self.min_delta = 1e-6

        if self.limit_to_stop < 0:
            raise ValueError("Stopper 'limit_to_stop' must be >= 0")
        if self.limit_warmup < 0:
            raise ValueError("Stopper 'limit_warmup' must be >= 0")
        if self.direction == "min":
            self.th, self.sign = float("inf"), 1
        elif self.direction == "max":
            self.th, self.sign = -float("inf"), -1
        else:
            raise ValueError("Stopper 'direction' must be 'min' or 'max'")

    def should_stop(self, current, current_metric):
        should_stop = False
        if current > self.limit_warmup:
            if self.sign * current_metric < self.sign * (
                (1 - self.min_delta) * self.th
            ):
                self.best_limit = current
                self.th = current_metric
            should_stop = (current - self.best_limit) >= self.limit_to_stop
        return should_stop
