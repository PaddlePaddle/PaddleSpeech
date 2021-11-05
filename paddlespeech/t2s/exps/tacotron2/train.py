# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import time
from collections import defaultdict

import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

from paddlespeech.t2s.data import dataset
from paddlespeech.t2s.exps.tacotron2.config import get_cfg_defaults
from paddlespeech.t2s.exps.tacotron2.ljspeech import LJSpeech
from paddlespeech.t2s.exps.tacotron2.ljspeech import LJSpeechCollector
from paddlespeech.t2s.models.tacotron2 import Tacotron2
from paddlespeech.t2s.models.tacotron2 import Tacotron2Loss
from paddlespeech.t2s.training.cli import default_argument_parser
from paddlespeech.t2s.training.experiment import ExperimentBase
from paddlespeech.t2s.utils import display
from paddlespeech.t2s.utils import mp_tools


class Experiment(ExperimentBase):
    def compute_losses(self, inputs, outputs):
        texts, mel_targets, plens, slens = inputs

        mel_outputs = outputs["mel_output"]
        mel_outputs_postnet = outputs["mel_outputs_postnet"]
        attention_weight = outputs["alignments"]
        if self.config.model.use_stop_token:
            stop_logits = outputs["stop_logits"]
        else:
            stop_logits = None

        losses = self.criterion(mel_outputs, mel_outputs_postnet, mel_targets,
                                attention_weight, slens, plens, stop_logits)
        return losses

    def train_batch(self):
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start

        self.optimizer.clear_grad()
        self.model.train()
        texts, mels, text_lens, output_lens = batch
        outputs = self.model(texts, text_lens, mels, output_lens)
        losses = self.compute_losses(batch, outputs)
        loss = losses["loss"]
        loss.backward()
        self.optimizer.step()
        iteration_time = time.time() - start

        losses_np = {k: float(v) for k, v in losses.items()}
        # logging
        msg = "Rank: {}, ".format(dist.get_rank())
        msg += "step: {}, ".format(self.iteration)
        msg += "time: {:>.3f}s/{:>.3f}s, ".format(data_loader_time,
                                                  iteration_time)
        msg += ', '.join('{}: {:>.6f}'.format(k, v)
                         for k, v in losses_np.items())
        self.logger.info(msg)

        if dist.get_rank() == 0:
            for k, v in losses_np.items():
                self.visualizer.add_scalar(f"train_loss/{k}", v, self.iteration)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        valid_losses = defaultdict(list)
        for i, batch in enumerate(self.valid_loader):
            texts, mels, text_lens, output_lens = batch
            outputs = self.model(texts, text_lens, mels, output_lens)
            losses = self.compute_losses(batch, outputs)
            for k, v in losses.items():
                valid_losses[k].append(float(v))

            attention_weights = outputs["alignments"]
            self.visualizer.add_figure(
                f"valid_sentence_{i}_alignments",
                display.plot_alignment(attention_weights[0].numpy().T),
                self.iteration)
            self.visualizer.add_figure(
                f"valid_sentence_{i}_target_spectrogram",
                display.plot_spectrogram(mels[0].numpy().T), self.iteration)
            self.visualizer.add_figure(
                f"valid_sentence_{i}_predicted_spectrogram",
                display.plot_spectrogram(outputs['mel_outputs_postnet'][0]
                                         .numpy().T), self.iteration)

        # write visual log
        valid_losses = {k: np.mean(v) for k, v in valid_losses.items()}

        # logging
        msg = "Valid: "
        msg += "step: {}, ".format(self.iteration)
        msg += ', '.join('{}: {:>.6f}'.format(k, v)
                         for k, v in valid_losses.items())
        self.logger.info(msg)

        for k, v in valid_losses.items():
            self.visualizer.add_scalar(f"valid/{k}", v, self.iteration)

    def setup_model(self):
        config = self.config
        model = Tacotron2(
            vocab_size=config.model.vocab_size,
            d_mels=config.data.n_mels,
            d_encoder=config.model.d_encoder,
            encoder_conv_layers=config.model.encoder_conv_layers,
            encoder_kernel_size=config.model.encoder_kernel_size,
            d_prenet=config.model.d_prenet,
            d_attention_rnn=config.model.d_attention_rnn,
            d_decoder_rnn=config.model.d_decoder_rnn,
            attention_filters=config.model.attention_filters,
            attention_kernel_size=config.model.attention_kernel_size,
            d_attention=config.model.d_attention,
            d_postnet=config.model.d_postnet,
            postnet_kernel_size=config.model.postnet_kernel_size,
            postnet_conv_layers=config.model.postnet_conv_layers,
            reduction_factor=config.model.reduction_factor,
            p_encoder_dropout=config.model.p_encoder_dropout,
            p_prenet_dropout=config.model.p_prenet_dropout,
            p_attention_dropout=config.model.p_attention_dropout,
            p_decoder_dropout=config.model.p_decoder_dropout,
            p_postnet_dropout=config.model.p_postnet_dropout,
            use_stop_token=config.model.use_stop_token)

        if self.parallel:
            model = paddle.DataParallel(model)

        grad_clip = paddle.nn.ClipGradByGlobalNorm(
            config.training.grad_clip_thresh)
        optimizer = paddle.optimizer.Adam(
            learning_rate=config.training.lr,
            parameters=model.parameters(),
            weight_decay=paddle.regularizer.L2Decay(
                config.training.weight_decay),
            grad_clip=grad_clip)
        criterion = Tacotron2Loss(
            use_stop_token_loss=config.model.use_stop_token,
            use_guided_attention_loss=config.model.use_guided_attention_loss,
            sigma=config.model.guided_attention_loss_sigma)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def setup_dataloader(self):
        args = self.args
        config = self.config
        ljspeech_dataset = LJSpeech(args.data)

        valid_set, train_set = dataset.split(ljspeech_dataset,
                                             config.data.valid_size)
        batch_fn = LJSpeechCollector(padding_idx=config.data.padding_idx)

        if not self.parallel:
            self.train_loader = DataLoader(
                train_set,
                batch_size=config.data.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=batch_fn)
        else:
            sampler = DistributedBatchSampler(
                train_set,
                batch_size=config.data.batch_size,
                shuffle=True,
                drop_last=True)
            self.train_loader = DataLoader(
                train_set, batch_sampler=sampler, collate_fn=batch_fn)

        self.valid_loader = DataLoader(
            valid_set,
            batch_size=config.data.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=batch_fn)


def main_sp(config, args):
    exp = Experiment(config, args)
    exp.setup()
    exp.resume_or_load()
    exp.run()


def main(config, args):
    if args.ngpu > 1:
        dist.spawn(main_sp, args=(config, args), nprocs=args.ngpu)
    else:
        main_sp(config, args)


if __name__ == "__main__":
    config = get_cfg_defaults()
    parser = default_argument_parser()
    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
