import os
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import jsonlines

from deepspeech.training.updaters.trainer import Trainer
from deepspeech.training.extensions import extension
from deepspeech.utils.mp_tools import rank_zero_only

from deepspeech.utils.log import Log

logger = Log(__name__).getlog()


def load_records(records_fp):
    """Load record files (json lines.)"""
    with jsonlines.open(records_fp, 'r') as reader:
        records = list(reader)
    return records


class Snapshot(extension.Extension):
    """An extension to make snapshot of the updater object inside
    the trainer. It is done by calling the updater's `save` method.
    An Updater save its state_dict by default, which contains the
    updater state, (i.e. epoch and iteration) and all the model
    parameters and optimizer states. If the updater inside the trainer
    subclasses StandardUpdater, everything is good to go.
    Parameters
    ----------
    checkpoint_dir : Union[str, Path]
        The directory to save checkpoints into.
    """

    trigger = (1, 'epoch')
    priority = -100
    default_name = "snapshot"

    def __init__(self, max_size: int=5, snapshot_on_error: bool=False):
        self.records: List[Dict[str, Any]] = []
        self.max_size = max_size
        self._snapshot_on_error = snapshot_on_error
        self._save_all = (max_size == -1)
        self.checkpoint_dir = None

    def initialize(self, trainer: Trainer):
        """Setting up this extention."""
        self.checkpoint_dir = trainer.out / "checkpoints"

        # load existing records
        record_path: Path = self.checkpoint_dir / "records.jsonl"
        if record_path.exists():
            logger.debug("Loading from an existing checkpoint dir")
            self.records = load_records(record_path)
            trainer.updater.load(self.records[-1]['path'])

    def on_error(self, trainer, exc, tb):
        if self._snapshot_on_error:
            self.save_checkpoint_and_update(trainer)

    def __call__(self, trainer: Trainer):
        self.save_checkpoint_and_update(trainer)

    def full(self):
        """Whether the number of snapshots it keeps track of is greater
        than the max_size."""
        return (not self._save_all) and len(self.records) > self.max_size

    @rank_zero_only
    def save_checkpoint_and_update(self, trainer: Trainer):
        """Saving new snapshot and remove the oldest snapshot if needed."""
        iteration = trainer.updater.state.iteration
        epoch = trainer.updater.state.epoch
        num = epoch if self.trigger[1] is 'epoch' else iteration
        path = self.checkpoint_dir / f"{num}.pdz"

        # add the new one
        trainer.updater.save(path)
        record = {
            "time": str(datetime.now()),
            'path': str(path.resolve()),  # use absolute path
            'iteration': iteration,
            'epoch': epoch,
        }
        self.records.append(record)

        # remove the earist
        if self.full():
            eariest_record = self.records[0]
            os.remove(eariest_record["path"])
            self.records.pop(0)

        # update the record file
        record_path = self.checkpoint_dir / "records.jsonl"
        with jsonlines.open(record_path, 'w') as writer:
            for record in self.records:
                # jsonlines.open may return a Writer or a Reader
                writer.write(record)  # pylint: disable=no-member