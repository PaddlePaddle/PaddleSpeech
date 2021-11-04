# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from chainer(https://github.com/chainer/chainer)
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import jsonlines

from paddlespeech.t2s.training import extension
from paddlespeech.t2s.training.trainer import Trainer
from paddlespeech.t2s.utils.mp_tools import rank_zero_only


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
            logging.debug("Loading from an existing checkpoint dir")
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
        path = self.checkpoint_dir / f"snapshot_iter_{iteration}.pdz"

        # add the new one
        trainer.updater.save(path)
        record = {
            "time": str(datetime.now()),
            'path': str(path.resolve()),  # use absolute path
            'iteration': iteration
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
