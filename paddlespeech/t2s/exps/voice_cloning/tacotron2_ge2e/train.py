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
from pathlib import Path

import numpy as np
import paddle
from matplotlib import pyplot as plt
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

from paddlespeech.t2s.data import dataset
from paddlespeech.t2s.exps.voice_cloning.tacotron2_ge2e.aishell3 import AiShell3
from paddlespeech.t2s.exps.voice_cloning.tacotron2_ge2e.aishell3 import collate_aishell3_examples
from paddlespeech.t2s.exps.voice_cloning.tacotron2_ge2e.config import get_cfg_defaults
from paddlespeech.t2s.models.tacotron2 import Tacotron2
from paddlespeech.t2s.models.tacotron2 import Tacotron2Loss
from paddlespeech.t2s.training.cli import default_argument_parser
from paddlespeech.t2s.training.experiment import ExperimentBase
from paddlespeech.t2s.utils import display
from paddlespeech.t2s.utils import mp_tools


class Experiment(ExperimentBase):
    def compute_losses(self, inputs, outputs):
        texts, tones, mel_targets, utterance_embeds, text_lens, output_lens, stop_tokens = inputs

        mel_outputs = outputs["mel_output"]
        mel_outputs_postnet = outputs["mel_outputs_postnet"]
        alignments = outputs["alignments"]

        losses = self.criterion(mel_outputs, mel_outputs_postnet, mel_targets,
                                alignments, output_lens, text_lens)
        return losses

    def train_batch(self):
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start

        self.optimizer.clear_grad()
        self.model.train()
        texts, tones, mels, utterance_embeds, text_lens, output_lens, stop_tokens = batch
        outputs = self.model(
            texts,
            text_lens,
            mels,
            output_lens,
            tones=tones,
            global_condition=utterance_embeds)
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
            for key, value in losses_np.items():
                self.visualizer.add_scalar(f"train_loss/{key}", value,
                                           self.iteration)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        valid_losses = defaultdict(list)
        for i, batch in enumerate(self.valid_loader):
            texts, tones, mels, utterance_embeds, text_lens, output_lens, stop_tokens = batch
            outputs = self.model(
                texts,
                text_lens,
                mels,
                output_lens,
                tones=tones,
                global_condition=utterance_embeds)
            losses = self.compute_losses(batch, outputs)
            for key, value in losses.items():
                valid_losses[key].append(float(value))

            attention_weights = outputs["alignments"]
            self.visualizer.add_figure(
                f"valid_sentence_{i}_alignments",
                display.plot_alignment(attention_weights[0].numpy().T),
                self.iteration)
            self.visualizer.add_figure(
                f"valid_sentence_{i}_target_spectrogram",
                display.plot_spectrogram(mels[0].numpy().T), self.iteration)
            mel_pred = outputs['mel_outputs_postnet']
            self.visualizer.add_figure(
                f"valid_sentence_{i}_predicted_spectrogram",
                display.plot_spectrogram(mel_pred[0].numpy().T), self.iteration)

        # write visual log
        valid_losses = {k: np.mean(v) for k, v in valid_losses.items()}

        # logging
        msg = "Valid: "
        msg += "step: {}, ".format(self.iteration)
        msg += ', '.join('{}: {:>.6f}'.format(k, v)
                         for k, v in valid_losses.items())
        self.logger.info(msg)

        for key, value in valid_losses.items():
            self.visualizer.add_scalar(f"valid/{key}", value, self.iteration)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def eval(self):
        """Evaluation of Tacotron2 in autoregressive manner."""
        self.model.eval()
        mel_dir = Path(self.output_dir / ("eval_{}".format(self.iteration)))
        mel_dir.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(self.test_loader):
            texts, tones, mels, utterance_embeds, *_ = batch
            outputs = self.model.infer(
                texts, tones=tones, global_condition=utterance_embeds)

            display.plot_alignment(outputs["alignments"][0].numpy().T)
            plt.savefig(mel_dir / f"sentence_{i}.png")
            plt.close()
            np.save(mel_dir / f"sentence_{i}",
                    outputs["mel_outputs_postnet"][0].numpy().T)
            print(f"sentence_{i}")

    def setup_model(self):
        config = self.config
        model = Tacotron2(
            vocab_size=config.model.vocab_size,
            n_tones=config.model.n_tones,
            d_mels=config.data.d_mels,
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
            d_global_condition=config.model.d_global_condition,
            use_stop_token=config.model.use_stop_token, )

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
        aishell3_dataset = AiShell3(args.data)

        valid_set, train_set = dataset.split(aishell3_dataset,
                                             config.data.valid_size)
        batch_fn = collate_aishell3_examples

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

        self.test_loader = DataLoader(
            valid_set,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=batch_fn)


def main_sp(config, args):
    exp = Experiment(config, args)
    exp.setup()
    exp.resume_or_load()
    if not args.test:
        exp.run()
    else:
        exp.eval()


def main(config, args):
    if args.ngpu > 1:
        dist.spawn(main_sp, args=(config, args), nprocs=args.ngpu)
    else:
        main_sp(config, args)


if __name__ == "__main__":
    config = get_cfg_defaults()
    parser = default_argument_parser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
