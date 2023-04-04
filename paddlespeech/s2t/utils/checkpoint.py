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
import glob
import json
import os
import re
from pathlib import Path
from typing import Text
from typing import Union

import paddle
from paddle import distributed as dist
from paddle.optimizer import Optimizer

from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ["Checkpoint"]


class Checkpoint():
    def __init__(self, kbest_n: int = 5, latest_n: int = 1):
        self.best_records: Mapping[Path, float] = {}
        self.latest_records = []
        self.kbest_n = kbest_n
        self.latest_n = latest_n
        self._save_all = (kbest_n == -1)

    def save_parameters(self,
                        checkpoint_dir,
                        tag_or_iteration: Union[int, Text],
                        model: paddle.nn.Layer,
                        optimizer: Optimizer = None,
                        infos: dict = None,
                        metric_type="val_loss"):
        """Save checkpoint in best_n and latest_n.

        Args:
            checkpoint_dir (str): the directory where checkpoint is saved.
            tag_or_iteration (int or str): the latest iteration(step or epoch) number or tag.
            model (Layer):  model to be checkpointed.
            optimizer (Optimizer, optional): optimizer to be checkpointed.
            infos (dict or None)):  any info you want to save.
            metric_type (str, optional): metric type. Defaults to "val_loss".
        """
        if (metric_type not in infos.keys()):
            self._save_parameters(checkpoint_dir, tag_or_iteration, model,
                                  optimizer, infos)
            return

        #save best
        if self._should_save_best(infos[metric_type]):
            self._save_best_checkpoint_and_update(infos[metric_type],
                                                  checkpoint_dir,
                                                  tag_or_iteration, model,
                                                  optimizer, infos)
        #save latest
        self._save_latest_checkpoint_and_update(checkpoint_dir,
                                                tag_or_iteration, model,
                                                optimizer, infos)

        if isinstance(tag_or_iteration, int):
            self._save_checkpoint_record(checkpoint_dir, tag_or_iteration)

    def load_parameters(self,
                        model,
                        optimizer=None,
                        checkpoint_dir=None,
                        checkpoint_path=None,
                        record_file="checkpoint_latest"):
        """Load a last model checkpoint from disk.
        Args:
            model (Layer): model to load parameters.
            optimizer (Optimizer, optional): optimizer to load states if needed.
                Defaults to None.
            checkpoint_dir (str, optional): the directory where checkpoint is saved.
            checkpoint_path (str, optional): if specified, load the checkpoint
                stored in the checkpoint_path(prefix) and the argument 'checkpoint_dir' will
                be ignored. Defaults to None.
            record_file "checkpoint_latest" or "checkpoint_best"
        Returns:
            configs (dict): epoch or step, lr and other meta info should be saved.
        """
        configs = {}

        if checkpoint_path:
            pass
        elif checkpoint_dir is not None and record_file is not None:
            # load checkpint from record file
            checkpoint_record = os.path.join(checkpoint_dir, record_file)
            iteration = self._load_checkpoint_idx(checkpoint_record)
            if iteration == -1:
                return configs
            checkpoint_path = os.path.join(checkpoint_dir,
                                           "{}".format(iteration))
        else:
            raise ValueError(
                "At least one of 'checkpoint_path' or 'checkpoint_dir' should be specified!"
            )

        rank = dist.get_rank()

        params_path = checkpoint_path + ".pdparams"
        model_dict = paddle.load(params_path)
        model.set_state_dict(model_dict)
        logger.info("Rank {}: Restore model from {}".format(rank, params_path))

        optimizer_path = checkpoint_path + ".pdopt"
        if optimizer and os.path.isfile(optimizer_path):
            optimizer_dict = paddle.load(optimizer_path)
            optimizer.set_state_dict(optimizer_dict)
            logger.info("Rank {}: Restore optimizer state from {}".format(
                rank, optimizer_path))

        info_path = re.sub('.pdparams$', '.json', params_path)
        if os.path.exists(info_path):
            with open(info_path, 'r') as fin:
                configs = json.load(fin)
        return configs

    def load_latest_parameters(self,
                               model,
                               optimizer=None,
                               checkpoint_dir=None,
                               checkpoint_path=None):
        """Load a last model checkpoint from disk.
        Args:
            model (Layer): model to load parameters.
            optimizer (Optimizer, optional): optimizer to load states if needed.
                Defaults to None.
            checkpoint_dir (str, optional): the directory where checkpoint is saved.
            checkpoint_path (str, optional): if specified, load the checkpoint
                stored in the checkpoint_path(prefix) and the argument 'checkpoint_dir' will
                be ignored. Defaults to None.
        Returns:
            configs (dict): epoch or step, lr and other meta info should be saved.
        """
        return self.load_parameters(model, optimizer, checkpoint_dir,
                                    checkpoint_path, "checkpoint_latest")

    def load_best_parameters(self,
                             model,
                             optimizer=None,
                             checkpoint_dir=None,
                             checkpoint_path=None):
        """Load a last model checkpoint from disk.
        Args:
            model (Layer): model to load parameters.
            optimizer (Optimizer, optional): optimizer to load states if needed.
                Defaults to None.
            checkpoint_dir (str, optional): the directory where checkpoint is saved.
            checkpoint_path (str, optional): if specified, load the checkpoint
                stored in the checkpoint_path(prefix) and the argument 'checkpoint_dir' will
                be ignored. Defaults to None.
        Returns:
            configs (dict): epoch or step, lr and other meta info should be saved.
        """
        return self.load_parameters(model, optimizer, checkpoint_dir,
                                    checkpoint_path, "checkpoint_best")

    def _should_save_best(self, metric: float) -> bool:
        if not self._best_full():
            return True

        # already full
        worst_record_path = max(self.best_records, key=self.best_records.get)
        # worst_record_path = max(self.best_records.iteritems(), key=operator.itemgetter(1))[0]
        worst_metric = self.best_records[worst_record_path]
        return metric < worst_metric

    def _best_full(self):
        return (not self._save_all) and len(self.best_records) == self.kbest_n

    def _latest_full(self):
        return len(self.latest_records) == self.latest_n

    def _save_best_checkpoint_and_update(self, metric, checkpoint_dir,
                                         tag_or_iteration, model, optimizer,
                                         infos):
        # remove the worst
        if self._best_full():
            worst_record_path = max(self.best_records,
                                    key=self.best_records.get)
            self.best_records.pop(worst_record_path)
            if (worst_record_path not in self.latest_records):
                logger.info(
                    "remove the worst checkpoint: {}".format(worst_record_path))
                self._del_checkpoint(checkpoint_dir, worst_record_path)

        # add the new one
        self._save_parameters(checkpoint_dir, tag_or_iteration, model,
                              optimizer, infos)
        self.best_records[tag_or_iteration] = metric

    def _save_latest_checkpoint_and_update(self, checkpoint_dir,
                                           tag_or_iteration, model, optimizer,
                                           infos):
        # remove the old
        if self._latest_full():
            to_del_fn = self.latest_records.pop(0)
            if (to_del_fn not in self.best_records.keys()):
                logger.info(
                    "remove the latest checkpoint: {}".format(to_del_fn))
                self._del_checkpoint(checkpoint_dir, to_del_fn)
        self.latest_records.append(tag_or_iteration)

        self._save_parameters(checkpoint_dir, tag_or_iteration, model,
                              optimizer, infos)

    def _del_checkpoint(self, checkpoint_dir, tag_or_iteration):
        checkpoint_path = os.path.join(checkpoint_dir,
                                       "{}".format(tag_or_iteration))
        for filename in glob.glob(checkpoint_path + ".*"):
            os.remove(filename)
            logger.info("delete file: {}".format(filename))

    def _load_checkpoint_idx(self, checkpoint_record: str) -> int:
        """Get the iteration number corresponding to the latest saved checkpoint.
        Args:
            checkpoint_path (str): the saved path of checkpoint.
        Returns:
            int: the latest iteration number. -1 for no checkpoint to load.
        """
        if not os.path.isfile(checkpoint_record):
            return -1

        # Fetch the latest checkpoint index.
        with open(checkpoint_record, "rt") as handle:
            latest_checkpoint = handle.readlines()[-1].strip()
            iteration = int(latest_checkpoint.split(":")[-1])
        return iteration

    def _save_checkpoint_record(self, checkpoint_dir: str, iteration: int):
        """Save the iteration number of the latest model to be checkpoint record.
        Args:
            checkpoint_dir (str): the directory where checkpoint is saved.
            iteration (int): the latest iteration number.
        Returns:
            None
        """
        checkpoint_record_latest = os.path.join(checkpoint_dir,
                                                "checkpoint_latest")
        checkpoint_record_best = os.path.join(checkpoint_dir, "checkpoint_best")

        with open(checkpoint_record_best, "w") as handle:
            for i in self.best_records.keys():
                handle.write("model_checkpoint_path:{}\n".format(i))
        with open(checkpoint_record_latest, "w") as handle:
            for i in self.latest_records:
                handle.write("model_checkpoint_path:{}\n".format(i))

    @mp_tools.rank_zero_only
    def _save_parameters(self,
                         checkpoint_dir: str,
                         tag_or_iteration: Union[int, str],
                         model: paddle.nn.Layer,
                         optimizer: Optimizer = None,
                         infos: dict = None):
        """Checkpoint the latest trained model parameters.
        Args:
            checkpoint_dir (str): the directory where checkpoint is saved.
            tag_or_iteration (int or str): the latest iteration(step or epoch) number.
            model (Layer): model to be checkpointed.
            optimizer (Optimizer, optional): optimizer to be checkpointed.
                Defaults to None.
            infos (dict or None): any info you want to save.
        Returns:
            None
        """
        checkpoint_path = os.path.join(checkpoint_dir,
                                       "{}".format(tag_or_iteration))

        model_dict = model.state_dict()
        params_path = checkpoint_path + ".pdparams"
        paddle.save(model_dict, params_path)
        logger.info("Saved model to {}".format(params_path))

        if optimizer:
            opt_dict = optimizer.state_dict()
            optimizer_path = checkpoint_path + ".pdopt"
            paddle.save(opt_dict, optimizer_path)
            logger.info("Saved optimzier state to {}".format(optimizer_path))

        info_path = re.sub('.pdparams$', '.json', params_path)
        infos = {} if infos is None else infos
        with open(info_path, 'w') as fout:
            data = json.dumps(infos)
            fout.write(data)
