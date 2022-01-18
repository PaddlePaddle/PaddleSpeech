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
from typing import Sequence

import paddle
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.nn import Layer
from paddle.optimizer import Optimizer

from paddlespeech.t2s.modules.losses import GuidedMultiHeadAttentionLoss
from paddlespeech.t2s.modules.losses import Tacotron2Loss as TransformerTTSLoss
from paddlespeech.t2s.training.extensions.evaluator import StandardEvaluator
from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.updaters.standard_updater import StandardUpdater
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TransformerTTSUpdater(StandardUpdater):
    def __init__(
            self,
            model: Layer,
            optimizer: Optimizer,
            dataloader: DataLoader,
            init_state=None,
            use_masking: bool=False,
            use_weighted_masking: bool=False,
            output_dir: Path=None,
            bce_pos_weight: float=5.0,
            loss_type: str="L1",
            use_guided_attn_loss: bool=True,
            modules_applied_guided_attn: Sequence[str]=("encoder-decoder"),
            guided_attn_loss_sigma: float=0.4,
            guided_attn_loss_lambda: float=1.0, ):
        super().__init__(model, optimizer, dataloader, init_state=None)

        self.loss_type = loss_type
        self.use_guided_attn_loss = use_guided_attn_loss
        self.modules_applied_guided_attn = modules_applied_guided_attn

        self.criterion = TransformerTTSLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight)

        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(
                sigma=guided_attn_loss_sigma,
                alpha=guided_attn_loss_lambda, )

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def update_core(self, batch):
        self.msg = "Rank: {}, ".format(dist.get_rank())
        losses_dict = {}

        after_outs, before_outs, logits, ys, stop_labels, olens, olens_in, need_dict = self.model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"], )

        l1_loss, l2_loss, bce_loss = self.criterion(
            after_outs=after_outs,
            before_outs=before_outs,
            logits=logits,
            ys=ys,
            stop_labels=stop_labels,
            olens=olens)

        report("train/bce_loss", float(bce_loss))
        report("train/l1_loss", float(l1_loss))
        report("train/l2_loss", float(l2_loss))
        losses_dict["bce_loss"] = float(bce_loss)
        losses_dict["l1_loss"] = float(l1_loss)
        losses_dict["l2_loss"] = float(l2_loss)
        # caluculate loss values
        if self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = l2_loss + bce_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss + bce_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)

        # calculate guided attention loss
        if self.use_guided_attn_loss:
            # calculate for encoder
            if "encoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                        reversed(range(len(need_dict['encoder'].encoders)))):
                    att_ws += [
                        need_dict['encoder'].encoders[layer_idx].self_attn.
                        attn[:, :need_dict['num_heads_applied_guided_attn']]
                    ]
                    if idx + 1 == need_dict['num_layers_applied_guided_attn']:
                        break
                # (B, H*L, T_in, T_in)
                att_ws = paddle.concat(att_ws, axis=1)
                enc_attn_loss = self.attn_criterion(
                    att_ws=att_ws,
                    ilens=batch["text_lengths"] + 1,
                    olens=batch["text_lengths"] + 1)
                loss = loss + enc_attn_loss
                report("train/enc_attn_loss", float(enc_attn_loss))
                losses_dict["enc_attn_loss"] = float(enc_attn_loss)
            # calculate for decoder
            if "decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                        reversed(range(len(need_dict['decoder'].decoders)))):
                    att_ws += [
                        need_dict['decoder'].decoders[layer_idx].self_attn.
                        attn[:, :need_dict['num_heads_applied_guided_attn']]
                    ]
                    if idx + 1 == need_dict['num_layers_applied_guided_attn']:
                        break
                # (B, H*L, T_out, T_out)
                att_ws = paddle.concat(att_ws, axis=1)
                dec_attn_loss = self.attn_criterion(
                    att_ws=att_ws, ilens=olens_in, olens=olens_in)
                report("train/dec_attn_loss", float(dec_attn_loss))
                losses_dict["dec_attn_loss"] = float(dec_attn_loss)
                loss = loss + dec_attn_loss
            # calculate for encoder-decoder
            if "encoder-decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                        reversed(range(len(need_dict['decoder'].decoders)))):
                    att_ws += [
                        need_dict['decoder'].decoders[layer_idx].src_attn.
                        attn[:, :need_dict['num_heads_applied_guided_attn']]
                    ]
                    if idx + 1 == need_dict['num_layers_applied_guided_attn']:
                        break
                # (B, H*L, T_out, T_in)
                att_ws = paddle.concat(att_ws, axis=1)
                enc_dec_attn_loss = self.attn_criterion(
                    att_ws=att_ws,
                    ilens=batch["text_lengths"] + 1,
                    olens=olens_in)
                report("train/enc_dec_attn_loss", float(enc_dec_attn_loss))
                losses_dict["enc_dec_attn_loss"] = float(enc_dec_attn_loss)
                loss = loss + enc_dec_attn_loss
        if need_dict['use_scaled_pos_enc']:
            report("train/encoder_alpha",
                   float(need_dict['encoder'].embed[-1].alpha))
            report("train/decoder_alpha",
                   float(need_dict['decoder'].embed[-1].alpha))
            losses_dict["encoder_alpha"] = float(
                need_dict['encoder'].embed[-1].alpha)
            losses_dict["decoder_alpha"] = float(
                need_dict['decoder'].embed[-1].alpha)

        optimizer = self.optimizer
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        report("train/loss", float(loss))
        losses_dict["loss"] = float(loss)
        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())


class TransformerTTSEvaluator(StandardEvaluator):
    def __init__(
            self,
            model: Layer,
            dataloader: DataLoader,
            init_state=None,
            use_masking: bool=False,
            use_weighted_masking: bool=False,
            output_dir: Path=None,
            bce_pos_weight: float=5.0,
            loss_type: str="L1",
            use_guided_attn_loss: bool=True,
            modules_applied_guided_attn: Sequence[str]=("encoder-decoder"),
            guided_attn_loss_sigma: float=0.4,
            guided_attn_loss_lambda: float=1.0, ):
        super().__init__(model, dataloader)

        self.loss_type = loss_type
        self.use_guided_attn_loss = use_guided_attn_loss
        self.modules_applied_guided_attn = modules_applied_guided_attn

        self.criterion = TransformerTTSLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight)

        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(
                sigma=guided_attn_loss_sigma,
                alpha=guided_attn_loss_lambda, )

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def evaluate_core(self, batch):
        self.msg = "Evaluate: "
        losses_dict = {}
        after_outs, before_outs, logits, ys, stop_labels, olens, olens_in, need_dict = self.model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"])

        l1_loss, l2_loss, bce_loss = self.criterion(
            after_outs=after_outs,
            before_outs=before_outs,
            logits=logits,
            ys=ys,
            stop_labels=stop_labels,
            olens=olens)

        report("eval/bce_loss", float(bce_loss))
        report("eval/l1_loss", float(l1_loss))
        report("eval/l2_loss", float(l2_loss))
        losses_dict["bce_loss"] = float(bce_loss)
        losses_dict["l1_loss"] = float(l1_loss)
        losses_dict["l2_loss"] = float(l2_loss)
        # caluculate loss values
        if self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = l2_loss + bce_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss + bce_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)

        # calculate guided attention loss
        if self.use_guided_attn_loss:
            # calculate for encoder
            if "encoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                        reversed(range(len(need_dict['encoder'].encoders)))):
                    att_ws += [
                        need_dict['encoder'].encoders[layer_idx].self_attn.
                        attn[:, :need_dict['num_heads_applied_guided_attn']]
                    ]
                    if idx + 1 == need_dict['num_layers_applied_guided_attn']:
                        break
                # (B, H*L, T_in, T_in)
                att_ws = paddle.concat(att_ws, axis=1)
                enc_attn_loss = self.attn_criterion(
                    att_ws=att_ws,
                    ilens=batch["text_lengths"] + 1,
                    olens=batch["text_lengths"] + 1)
                loss = loss + enc_attn_loss
                report("train/enc_attn_loss", float(enc_attn_loss))
                losses_dict["enc_attn_loss"] = float(enc_attn_loss)
            # calculate for decoder
            if "decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                        reversed(range(len(need_dict['decoder'].decoders)))):
                    att_ws += [
                        need_dict['decoder'].decoders[layer_idx].self_attn.
                        attn[:, :need_dict['num_heads_applied_guided_attn']]
                    ]
                    if idx + 1 == need_dict['num_layers_applied_guided_attn']:
                        break
                # (B, H*L, T_out, T_out)
                att_ws = paddle.concat(att_ws, axis=1)
                dec_attn_loss = self.attn_criterion(
                    att_ws=att_ws, ilens=olens_in, olens=olens_in)
                report("eval/dec_attn_loss", float(dec_attn_loss))
                losses_dict["dec_attn_loss"] = float(dec_attn_loss)
                loss = loss + dec_attn_loss
            # calculate for encoder-decoder
            if "encoder-decoder" in self.modules_applied_guided_attn:

                att_ws = []
                for idx, layer_idx in enumerate(
                        reversed(range(len(need_dict['decoder'].decoders)))):
                    att_ws += [
                        need_dict['decoder'].decoders[layer_idx].src_attn.
                        attn[:, :need_dict['num_heads_applied_guided_attn']]
                    ]
                    if idx + 1 == need_dict['num_layers_applied_guided_attn']:
                        break
                # (B, H*L, T_out, T_in)
                att_ws = paddle.concat(att_ws, axis=1)
                enc_dec_attn_loss = self.attn_criterion(
                    att_ws=att_ws,
                    ilens=batch["text_lengths"] + 1,
                    olens=olens_in)
                report("eval/enc_dec_attn_loss", float(enc_dec_attn_loss))
                losses_dict["enc_dec_attn_loss"] = float(enc_dec_attn_loss)
                loss = loss + enc_dec_attn_loss
        if need_dict['use_scaled_pos_enc']:
            report("eval/encoder_alpha",
                   float(need_dict['encoder'].embed[-1].alpha))
            report("eval/decoder_alpha",
                   float(need_dict['decoder'].embed[-1].alpha))
            losses_dict["encoder_alpha"] = float(
                need_dict['encoder'].embed[-1].alpha)
            losses_dict["decoder_alpha"] = float(
                need_dict['decoder'].embed[-1].alpha)
        report("eval/loss", float(loss))
        losses_dict["loss"] = float(loss)
        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        self.logger.info(self.msg)
