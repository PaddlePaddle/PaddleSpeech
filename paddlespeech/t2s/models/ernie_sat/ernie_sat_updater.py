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
import logging
from pathlib import Path

from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.nn import Layer
from paddle.optimizer import Optimizer

from paddlespeech.t2s.modules.losses import MLMLoss
from paddlespeech.t2s.training.extensions.evaluator import StandardEvaluator
from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.updaters.standard_updater import StandardUpdater
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ErnieSATUpdater(StandardUpdater):
    def __init__(self,
                 model: Layer,
                 optimizer: Optimizer,
                 dataloader: DataLoader,
                 init_state=None,
                 text_masking: bool=False,
                 odim: int=80,
                 output_dir: Path=None):
        super().__init__(model, optimizer, dataloader, init_state=None)

        self.criterion = MLMLoss(text_masking=text_masking, odim=odim)

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def update_core(self, batch):
        self.msg = "Rank: {}, ".format(dist.get_rank())
        losses_dict = {}

        before_outs, after_outs, text_outs = self.model(
            speech=batch["speech"],
            text=batch["text"],
            masked_pos=batch["masked_pos"],
            speech_mask=batch["speech_mask"],
            text_mask=batch["text_mask"],
            speech_seg_pos=batch["speech_seg_pos"],
            text_seg_pos=batch["text_seg_pos"])

        mlm_loss, text_mlm_loss = self.criterion(
            speech=batch["speech"],
            before_outs=before_outs,
            after_outs=after_outs,
            masked_pos=batch["masked_pos"],
            text=batch["text"],
            # maybe None
            text_outs=text_outs,
            # maybe None
            text_masked_pos=batch["text_masked_pos"])

        loss = mlm_loss + text_mlm_loss if text_mlm_loss is not None else mlm_loss

        optimizer = self.optimizer
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        report("train/loss", float(loss))
        report("train/mlm_loss", float(mlm_loss))
        if text_mlm_loss is not None:
            report("train/text_mlm_loss", float(text_mlm_loss))
            losses_dict["text_mlm_loss"] = float(text_mlm_loss)

        losses_dict["mlm_loss"] = float(mlm_loss)
        losses_dict["loss"] = float(loss)
        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())


class ErnieSATEvaluator(StandardEvaluator):
    def __init__(self,
                 model: Layer,
                 dataloader: DataLoader,
                 text_masking: bool=False,
                 odim: int=80,
                 output_dir: Path=None):
        super().__init__(model, dataloader)

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

        self.criterion = MLMLoss(text_masking=text_masking, odim=odim)

    def evaluate_core(self, batch):
        self.msg = "Evaluate: "
        losses_dict = {}

        before_outs, after_outs, text_outs = self.model(
            speech=batch["speech"],
            text=batch["text"],
            masked_pos=batch["masked_pos"],
            speech_mask=batch["speech_mask"],
            text_mask=batch["text_mask"],
            speech_seg_pos=batch["speech_seg_pos"],
            text_seg_pos=batch["text_seg_pos"])

        mlm_loss, text_mlm_loss = self.criterion(
            speech=batch["speech"],
            before_outs=before_outs,
            after_outs=after_outs,
            masked_pos=batch["masked_pos"],
            text=batch["text"],
            # maybe None
            text_outs=text_outs,
            # maybe None
            text_masked_pos=batch["text_masked_pos"])
        loss = mlm_loss + text_mlm_loss if text_mlm_loss is not None else mlm_loss

        report("eval/loss", float(loss))
        report("eval/mlm_loss", float(mlm_loss))
        if text_mlm_loss is not None:
            report("eval/text_mlm_loss", float(text_mlm_loss))
            losses_dict["text_mlm_loss"] = float(text_mlm_loss)

        losses_dict["mlm_loss"] = float(mlm_loss)
        losses_dict["loss"] = float(loss)

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        self.logger.info(self.msg)
