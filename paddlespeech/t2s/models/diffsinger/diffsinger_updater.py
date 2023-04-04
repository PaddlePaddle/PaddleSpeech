# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Dict

import paddle
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.nn import Layer
from paddle.optimizer import Optimizer

from paddlespeech.t2s.training.extensions.evaluator import StandardEvaluator
from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.updaters.standard_updater import StandardUpdater
from paddlespeech.t2s.training.updaters.standard_updater import UpdaterState

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DiffSingerUpdater(StandardUpdater):
    def __init__(self,
                 model: Layer,
                 optimizers: Dict[str, Optimizer],
                 criterions: Dict[str, Layer],
                 dataloader: DataLoader,
                 ds_train_start_steps: int = 160000,
                 output_dir: Path = None,
                 only_train_diffusion: bool = True):
        super().__init__(model, optimizers, dataloader, init_state=None)
        self.model = model._layers if isinstance(model,
                                                 paddle.DataParallel) else model
        self.only_train_diffusion = only_train_diffusion

        self.optimizers = optimizers
        self.optimizer_fs2: Optimizer = optimizers['fs2']
        self.optimizer_ds: Optimizer = optimizers['ds']

        self.criterions = criterions
        self.criterion_fs2 = criterions['fs2']
        self.criterion_ds = criterions['ds']

        self.dataloader = dataloader

        self.ds_train_start_steps = ds_train_start_steps

        self.state = UpdaterState(iteration=0, epoch=0)
        self.train_iterator = iter(self.dataloader)

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def update_core(self, batch):
        self.msg = "Rank: {}, ".format(dist.get_rank())
        losses_dict = {}
        # spk_id!=None in multiple spk diffsinger
        spk_id = batch["spk_id"] if "spk_id" in batch else None
        spk_emb = batch["spk_emb"] if "spk_emb" in batch else None
        # No explicit speaker identifier labels are used during voice cloning training.
        if spk_emb is not None:
            spk_id = None

        # only train fastspeech2 module firstly
        if self.state.iteration < self.ds_train_start_steps:
            before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens, spk_logits = self.model(
                text=batch["text"],
                note=batch["note"],
                note_dur=batch["note_dur"],
                is_slur=batch["is_slur"],
                text_lengths=batch["text_lengths"],
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
                durations=batch["durations"],
                pitch=batch["pitch"],
                energy=batch["energy"],
                spk_id=spk_id,
                spk_emb=spk_emb,
                only_train_fs2=True,
            )

            l1_loss_fs2, ssim_loss_fs2, duration_loss, pitch_loss, energy_loss, speaker_loss = self.criterion_fs2(
                after_outs=after_outs,
                before_outs=before_outs,
                d_outs=d_outs,
                p_outs=p_outs,
                e_outs=e_outs,
                ys=ys,
                ds=batch["durations"],
                ps=batch["pitch"],
                es=batch["energy"],
                ilens=batch["text_lengths"],
                olens=olens,
                spk_logits=spk_logits,
                spk_ids=spk_id,
            )

            loss_fs2 = l1_loss_fs2 + ssim_loss_fs2 + duration_loss + pitch_loss + energy_loss + speaker_loss

            self.optimizer_fs2.clear_grad()
            loss_fs2.backward()
            self.optimizer_fs2.step()

            report("train/loss_fs2", float(loss_fs2))
            report("train/l1_loss_fs2", float(l1_loss_fs2))
            report("train/ssim_loss_fs2", float(ssim_loss_fs2))
            report("train/duration_loss", float(duration_loss))
            report("train/pitch_loss", float(pitch_loss))

            losses_dict["l1_loss_fs2"] = float(l1_loss_fs2)
            losses_dict["ssim_loss_fs2"] = float(ssim_loss_fs2)
            losses_dict["duration_loss"] = float(duration_loss)
            losses_dict["pitch_loss"] = float(pitch_loss)

            if speaker_loss != 0.:
                report("train/speaker_loss", float(speaker_loss))
                losses_dict["speaker_loss"] = float(speaker_loss)
            if energy_loss != 0.:
                report("train/energy_loss", float(energy_loss))
                losses_dict["energy_loss"] = float(energy_loss)

            losses_dict["loss_fs2"] = float(loss_fs2)
            self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                                  for k, v in losses_dict.items())

        # Then only train diffusion module, freeze fastspeech2 parameters.
        if self.state.iteration > self.ds_train_start_steps:
            for param in self.model.fs2.parameters():
                param.trainable = False if self.only_train_diffusion else True

            noise_pred, noise_target, mel_masks = self.model(
                text=batch["text"],
                note=batch["note"],
                note_dur=batch["note_dur"],
                is_slur=batch["is_slur"],
                text_lengths=batch["text_lengths"],
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
                durations=batch["durations"],
                pitch=batch["pitch"],
                energy=batch["energy"],
                spk_id=spk_id,
                spk_emb=spk_emb,
                only_train_fs2=False,
            )

            noise_pred = noise_pred.transpose((0, 2, 1))
            noise_target = noise_target.transpose((0, 2, 1))
            mel_masks = mel_masks.transpose((0, 2, 1))
            l1_loss_ds = self.criterion_ds(
                noise_pred=noise_pred,
                noise_target=noise_target,
                mel_masks=mel_masks,
            )

            loss_ds = l1_loss_ds

            self.optimizer_ds.clear_grad()
            loss_ds.backward()
            self.optimizer_ds.step()

            report("train/loss_ds", float(loss_ds))
            report("train/l1_loss_ds", float(l1_loss_ds))
            losses_dict["l1_loss_ds"] = float(l1_loss_ds)
            losses_dict["loss_ds"] = float(loss_ds)
            self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                                  for k, v in losses_dict.items())

        self.logger.info(self.msg)


class DiffSingerEvaluator(StandardEvaluator):
    def __init__(
        self,
        model: Layer,
        criterions: Dict[str, Layer],
        dataloader: DataLoader,
        output_dir: Path = None,
    ):
        super().__init__(model, dataloader)
        self.model = model._layers if isinstance(model,
                                                 paddle.DataParallel) else model

        self.criterions = criterions
        self.criterion_fs2 = criterions['fs2']
        self.criterion_ds = criterions['ds']
        self.dataloader = dataloader

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def evaluate_core(self, batch):
        self.msg = "Evaluate: "
        losses_dict = {}
        # spk_id!=None in multiple spk diffsinger
        spk_id = batch["spk_id"] if "spk_id" in batch else None
        spk_emb = batch["spk_emb"] if "spk_emb" in batch else None
        if spk_emb is not None:
            spk_id = None

        # Here show fastspeech2 eval
        before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens, spk_logits = self.model(
            text=batch["text"],
            note=batch["note"],
            note_dur=batch["note_dur"],
            is_slur=batch["is_slur"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"],
            durations=batch["durations"],
            pitch=batch["pitch"],
            energy=batch["energy"],
            spk_id=spk_id,
            spk_emb=spk_emb,
            only_train_fs2=True,
        )

        l1_loss_fs2, ssim_loss_fs2, duration_loss, pitch_loss, energy_loss, speaker_loss = self.criterion_fs2(
            after_outs=after_outs,
            before_outs=before_outs,
            d_outs=d_outs,
            p_outs=p_outs,
            e_outs=e_outs,
            ys=ys,
            ds=batch["durations"],
            ps=batch["pitch"],
            es=batch["energy"],
            ilens=batch["text_lengths"],
            olens=olens,
            spk_logits=spk_logits,
            spk_ids=spk_id,
        )

        loss_fs2 = l1_loss_fs2 + ssim_loss_fs2 + duration_loss + pitch_loss + energy_loss + speaker_loss

        report("eval/loss_fs2", float(loss_fs2))
        report("eval/l1_loss_fs2", float(l1_loss_fs2))
        report("eval/ssim_loss_fs2", float(ssim_loss_fs2))
        report("eval/duration_loss", float(duration_loss))
        report("eval/pitch_loss", float(pitch_loss))

        losses_dict["l1_loss_fs2"] = float(l1_loss_fs2)
        losses_dict["ssim_loss_fs2"] = float(ssim_loss_fs2)
        losses_dict["duration_loss"] = float(duration_loss)
        losses_dict["pitch_loss"] = float(pitch_loss)

        if speaker_loss != 0.:
            report("eval/speaker_loss", float(speaker_loss))
            losses_dict["speaker_loss"] = float(speaker_loss)
        if energy_loss != 0.:
            report("eval/energy_loss", float(energy_loss))
            losses_dict["energy_loss"] = float(energy_loss)

        losses_dict["loss_fs2"] = float(loss_fs2)

        # Here show diffusion eval
        noise_pred, noise_target, mel_masks = self.model(
            text=batch["text"],
            note=batch["note"],
            note_dur=batch["note_dur"],
            is_slur=batch["is_slur"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"],
            durations=batch["durations"],
            pitch=batch["pitch"],
            energy=batch["energy"],
            spk_id=spk_id,
            spk_emb=spk_emb,
            only_train_fs2=False,
        )

        noise_pred = noise_pred.transpose((0, 2, 1))
        noise_target = noise_target.transpose((0, 2, 1))
        mel_masks = mel_masks.transpose((0, 2, 1))
        l1_loss_ds = self.criterion_ds(
            noise_pred=noise_pred,
            noise_target=noise_target,
            mel_masks=mel_masks,
        )

        loss_ds = l1_loss_ds

        report("eval/loss_ds", float(loss_ds))
        report("eval/l1_loss_ds", float(l1_loss_ds))
        losses_dict["l1_loss_ds"] = float(l1_loss_ds)
        losses_dict["loss_ds"] = float(loss_ds)
        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())

        self.logger.info(self.msg)
