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

from paddle import distributed as dist

from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Loss
from paddlespeech.t2s.training.extensions.evaluator import StandardEvaluator
from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.updaters.standard_updater import StandardUpdater
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FastSpeech2Updater(StandardUpdater):
    def __init__(self,
                 model,
                 optimizer,
                 dataloader,
                 init_state=None,
                 use_masking=False,
                 use_weighted_masking=False,
                 output_dir=None):
        super().__init__(model, optimizer, dataloader, init_state=None)
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        self.criterion = FastSpeech2Loss(
            use_masking=self.use_masking,
            use_weighted_masking=self.use_weighted_masking)

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
        # No explicit speaker identifier labels are used during voice cloning training.
        if spk_emb is not None:
            spk_id = None

        before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens = self.model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"],
            durations=batch["durations"],
            pitch=batch["pitch"],
            energy=batch["energy"],
            spk_id=spk_id,
            spk_emb=spk_emb)

        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
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
            olens=olens)

        loss = l1_loss + duration_loss + pitch_loss + energy_loss

        optimizer = self.optimizer
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        report("train/loss", float(loss))
        report("train/l1_loss", float(l1_loss))
        report("train/duration_loss", float(duration_loss))
        report("train/pitch_loss", float(pitch_loss))
        report("train/energy_loss", float(energy_loss))

        losses_dict["l1_loss"] = float(l1_loss)
        losses_dict["duration_loss"] = float(duration_loss)
        losses_dict["pitch_loss"] = float(pitch_loss)
        losses_dict["energy_loss"] = float(energy_loss)
        losses_dict["loss"] = float(loss)
        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())


class FastSpeech2Evaluator(StandardEvaluator):
    def __init__(self,
                 model,
                 dataloader,
                 use_masking=False,
                 use_weighted_masking=False,
                 output_dir=None):
        super().__init__(model, dataloader)
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

        self.criterion = FastSpeech2Loss(
            use_masking=self.use_masking,
            use_weighted_masking=self.use_weighted_masking)

    def evaluate_core(self, batch):
        self.msg = "Evaluate: "
        losses_dict = {}
        # spk_id!=None in multiple spk fastspeech2 
        spk_id = batch["spk_id"] if "spk_id" in batch else None
        spk_emb = batch["spk_emb"] if "spk_emb" in batch else None
        if spk_emb is not None:
            spk_id = None

        before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens = self.model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"],
            durations=batch["durations"],
            pitch=batch["pitch"],
            energy=batch["energy"],
            spk_id=spk_id,
            spk_emb=spk_emb)

        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
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
            olens=olens, )
        loss = l1_loss + duration_loss + pitch_loss + energy_loss

        report("eval/loss", float(loss))
        report("eval/l1_loss", float(l1_loss))
        report("eval/duration_loss", float(duration_loss))
        report("eval/pitch_loss", float(pitch_loss))
        report("eval/energy_loss", float(energy_loss))

        losses_dict["l1_loss"] = float(l1_loss)
        losses_dict["duration_loss"] = float(duration_loss)
        losses_dict["pitch_loss"] = float(pitch_loss)
        losses_dict["energy_loss"] = float(energy_loss)
        losses_dict["loss"] = float(loss)
        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        self.logger.info(self.msg)
