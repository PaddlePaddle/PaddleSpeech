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

from paddlespeech.t2s.modules.losses import GuidedAttentionLoss
from paddlespeech.t2s.modules.losses import Tacotron2Loss
from paddlespeech.t2s.training.extensions.evaluator import StandardEvaluator
from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.updaters.standard_updater import StandardUpdater

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Tacotron2Updater(StandardUpdater):
    def __init__(self,
                 model: Layer,
                 optimizer: Optimizer,
                 dataloader: DataLoader,
                 init_state=None,
                 use_masking: bool = True,
                 use_weighted_masking: bool = False,
                 bce_pos_weight: float = 5.0,
                 loss_type: str = "L1+L2",
                 use_guided_attn_loss: bool = True,
                 guided_attn_loss_sigma: float = 0.4,
                 guided_attn_loss_lambda: float = 1.0,
                 output_dir: Path = None):
        super().__init__(model, optimizer, dataloader, init_state=None)

        self.loss_type = loss_type
        self.use_guided_attn_loss = use_guided_attn_loss

        self.taco2_loss = Tacotron2Loss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight,
        )
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(
                sigma=guided_attn_loss_sigma,
                alpha=guided_attn_loss_lambda,
            )

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def update_core(self, batch):
        self.msg = "Rank: {}, ".format(dist.get_rank())
        losses_dict = {}
        # spk_id!=None in multiple spk fastspeech2
        spk_id = batch["spk_id"] if "spk_id" in batch else None
        spk_emb = batch["spk_emb"] if "spk_emb" in batch else None
        if spk_emb is not None:
            spk_id = None

        after_outs, before_outs, logits, ys, stop_labels, olens, att_ws, olens_in = self.model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"],
            spk_id=spk_id,
            spk_emb=spk_emb)

        # calculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(after_outs=after_outs,
                                                      before_outs=before_outs,
                                                      logits=logits,
                                                      ys=ys,
                                                      stop_labels=stop_labels,
                                                      olens=olens)

        if self.loss_type == "L1+L2":
            loss = l1_loss + mse_loss + bce_loss
        elif self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = mse_loss + bce_loss
        else:
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        # calculate attention loss
        if self.use_guided_attn_loss:
            # NOTE: length of output for auto-regressive
            # input will be changed when r > 1
            attn_loss = self.attn_loss(att_ws=att_ws,
                                       ilens=batch["text_lengths"] + 1,
                                       olens=olens_in)
            loss = loss + attn_loss

        optimizer = self.optimizer
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        if self.use_guided_attn_loss:
            report("train/attn_loss", float(attn_loss))
            losses_dict["attn_loss"] = float(attn_loss)

        report("train/l1_loss", float(l1_loss))
        report("train/mse_loss", float(mse_loss))
        report("train/bce_loss", float(bce_loss))
        report("train/loss", float(loss))

        losses_dict["l1_loss"] = float(l1_loss)
        losses_dict["mse_loss"] = float(mse_loss)
        losses_dict["bce_loss"] = float(bce_loss)
        losses_dict["loss"] = float(loss)
        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())


class Tacotron2Evaluator(StandardEvaluator):
    def __init__(self,
                 model: Layer,
                 dataloader: DataLoader,
                 use_masking: bool = True,
                 use_weighted_masking: bool = False,
                 bce_pos_weight: float = 5.0,
                 loss_type: str = "L1+L2",
                 use_guided_attn_loss: bool = True,
                 guided_attn_loss_sigma: float = 0.4,
                 guided_attn_loss_lambda: float = 1.0,
                 output_dir=None):
        super().__init__(model, dataloader)

        self.loss_type = loss_type
        self.use_guided_attn_loss = use_guided_attn_loss

        self.taco2_loss = Tacotron2Loss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight,
        )
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(
                sigma=guided_attn_loss_sigma,
                alpha=guided_attn_loss_lambda,
            )

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def evaluate_core(self, batch):
        self.msg = "Evaluate: "
        losses_dict = {}
        # spk_id!=None in multiple spk fastspeech2
        spk_id = batch["spk_id"] if "spk_id" in batch else None
        spk_emb = batch["spk_emb"] if "spk_emb" in batch else None
        if spk_emb is not None:
            spk_id = None

        after_outs, before_outs, logits, ys, stop_labels, olens, att_ws, olens_in = self.model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"],
            spk_id=spk_id,
            spk_emb=spk_emb)

        # calculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(after_outs=after_outs,
                                                      before_outs=before_outs,
                                                      logits=logits,
                                                      ys=ys,
                                                      stop_labels=stop_labels,
                                                      olens=olens)

        if self.loss_type == "L1+L2":
            loss = l1_loss + mse_loss + bce_loss
        elif self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = mse_loss + bce_loss
        else:
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        # calculate attention loss
        if self.use_guided_attn_loss:
            # NOTE: length of output for auto-regressive
            # input will be changed when r > 1
            attn_loss = self.attn_loss(att_ws=att_ws,
                                       ilens=batch["text_lengths"] + 1,
                                       olens=olens_in)
            loss = loss + attn_loss

        if self.use_guided_attn_loss:
            report("eval/attn_loss", float(attn_loss))
            losses_dict["attn_loss"] = float(attn_loss)

        report("eval/l1_loss", float(l1_loss))
        report("eval/mse_loss", float(mse_loss))
        report("eval/bce_loss", float(bce_loss))
        report("eval/loss", float(loss))

        losses_dict["l1_loss"] = float(l1_loss)
        losses_dict["mse_loss"] = float(mse_loss)
        losses_dict["bce_loss"] = float(bce_loss)
        losses_dict["loss"] = float(loss)
        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        self.logger.info(self.msg)
