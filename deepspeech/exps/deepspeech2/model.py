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
"""Contains DeepSpeech2 model."""
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader
from yacs.config import CfgNode

from deepspeech.io.collator import SpeechCollator
from deepspeech.io.dataset import ManifestDataset
from deepspeech.io.sampler import SortagradBatchSampler
from deepspeech.io.sampler import SortagradDistributedBatchSampler
from deepspeech.models.deepspeech2 import DeepSpeech2InferModel
from deepspeech.models.deepspeech2 import DeepSpeech2Model
from deepspeech.training.gradclip import ClipGradByGlobalNormWithLog
from deepspeech.training.trainer import Trainer
from deepspeech.utils import error_rate
from deepspeech.utils import layer_tools
from deepspeech.utils import mp_tools
from deepspeech.utils.log import Log
logger = Log(__name__).getlog()


class DeepSpeech2Trainer(Trainer):
    @classmethod
    def params(cls, config: Optional[CfgNode]=None) -> CfgNode:
        # training config
        default = CfgNode(
            dict(
                lr=5e-4,  # learning rate
                lr_decay=1.0,  # learning rate decay
                weight_decay=1e-6,  # the coeff of weight decay
                global_grad_clip=5.0,  # the global norm clip
                n_epoch=50,  # train epochs
            ))

        if config is not None:
            config.merge_from_other_cfg(default)
        return default

    def __init__(self, config, args):
        super().__init__(config, args)

    def train_batch(self, batch_index, batch_data, msg):
        start = time.time()
        utt, audio, audio_len, text, text_len = batch_data
        loss = self.model(audio, audio_len, text, text_len)
        loss.backward()
        layer_tools.print_grads(self.model, print_func=None)
        self.optimizer.step()
        self.optimizer.clear_grad()
        iteration_time = time.time() - start

        losses_np = {
            'train_loss': float(loss),
        }
        msg += "train time: {:>.3f}s, ".format(iteration_time)
        msg += "batch size: {}, ".format(self.config.collator.batch_size)
        msg += ', '.join('{}: {:>.6f}'.format(k, v)
                         for k, v in losses_np.items())
        logger.info(msg)

        if dist.get_rank() == 0 and self.visualizer:
            for k, v in losses_np.items():
                self.visualizer.add_scalar("train/{}".format(k), v,
                                           self.iteration)
        self.iteration += 1

    @paddle.no_grad()
    def valid(self):
        logger.info(f"Valid Total Examples: {len(self.valid_loader.dataset)}")
        self.model.eval()
        valid_losses = defaultdict(list)
        num_seen_utts = 1
        total_loss = 0.0
        for i, batch in enumerate(self.valid_loader):
            utt, audio, audio_len, text, text_len = batch
            loss = self.model(audio, audio_len, text, text_len)
            if paddle.isfinite(loss):
                num_utts = batch[1].shape[0]
                num_seen_utts += num_utts
                total_loss += float(loss) * num_utts
                valid_losses['val_loss'].append(float(loss))

            if (i + 1) % self.config.training.log_interval == 0:
                valid_dump = {k: np.mean(v) for k, v in valid_losses.items()}
                valid_dump['val_history_loss'] = total_loss / num_seen_utts

                # logging
                msg = f"Valid: Rank: {dist.get_rank()}, "
                msg += "epoch: {}, ".format(self.epoch)
                msg += "step: {}, ".format(self.iteration)
                msg += "batch : {}/{}, ".format(i + 1, len(self.valid_loader))
                msg += ', '.join('{}: {:>.6f}'.format(k, v)
                                 for k, v in valid_dump.items())
                logger.info(msg)

        logger.info('Rank {} Val info val_loss {}'.format(
            dist.get_rank(), total_loss / num_seen_utts))
        return total_loss, num_seen_utts

    def setup_model(self):
        config = self.config
        model = DeepSpeech2Model(
            feat_size=self.train_loader.collate_fn.feature_size,
            dict_size=self.train_loader.collate_fn.vocab_size,
            num_conv_layers=config.model.num_conv_layers,
            num_rnn_layers=config.model.num_rnn_layers,
            rnn_size=config.model.rnn_layer_size,
            use_gru=config.model.use_gru,
            share_rnn_weights=config.model.share_rnn_weights)

        if self.parallel:
            model = paddle.DataParallel(model)

        logger.info(f"{model}")
        layer_tools.print_params(model, logger.info)

        grad_clip = ClipGradByGlobalNormWithLog(
            config.training.global_grad_clip)
        lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=config.training.lr,
            gamma=config.training.lr_decay,
            verbose=True)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=paddle.regularizer.L2Decay(
                config.training.weight_decay),
            grad_clip=grad_clip)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        logger.info("Setup model/optimizer/lr_scheduler!")

    def setup_dataloader(self):
        config = self.config.clone()
        config.defrost()
        config.collator.keep_transcription_text = False

        config.data.manifest = config.data.train_manifest
        train_dataset = ManifestDataset.from_config(config)

        config.data.manifest = config.data.dev_manifest
        dev_dataset = ManifestDataset.from_config(config)

        if self.parallel:
            batch_sampler = SortagradDistributedBatchSampler(
                train_dataset,
                batch_size=config.collator.batch_size,
                num_replicas=None,
                rank=None,
                shuffle=True,
                drop_last=True,
                sortagrad=config.collator.sortagrad,
                shuffle_method=config.collator.shuffle_method)
        else:
            batch_sampler = SortagradBatchSampler(
                train_dataset,
                shuffle=True,
                batch_size=config.collator.batch_size,
                drop_last=True,
                sortagrad=config.collator.sortagrad,
                shuffle_method=config.collator.shuffle_method)

        collate_fn_train = SpeechCollator.from_config(config)

        config.collator.augmentation_config = ""
        collate_fn_dev = SpeechCollator.from_config(config)
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn_train,
            num_workers=config.collator.num_workers)
        self.valid_loader = DataLoader(
            dev_dataset,
            batch_size=config.collator.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn_dev)
        logger.info("Setup train/valid Dataloader!")


class DeepSpeech2Tester(DeepSpeech2Trainer):
    @classmethod
    def params(cls, config: Optional[CfgNode]=None) -> CfgNode:
        # testing config
        default = CfgNode(
            dict(
                alpha=2.5,  # Coef of LM for beam search.
                beta=0.3,  # Coef of WC for beam search.
                cutoff_prob=1.0,  # Cutoff probability for pruning.
                cutoff_top_n=40,  # Cutoff number for pruning.
                lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm',  # Filepath for language model.
                decoding_method='ctc_beam_search',  # Decoding method. Options: ctc_beam_search, ctc_greedy
                error_rate_type='wer',  # Error rate type for evaluation. Options `wer`, 'cer'
                num_proc_bsearch=8,  # # of CPUs for beam search.
                beam_size=500,  # Beam search width.
                batch_size=128,  # decoding batch size
            ))

        if config is not None:
            config.merge_from_other_cfg(default)
        return default

    def __init__(self, config, args):
        super().__init__(config, args)

    def ordid2token(self, texts, texts_len):
        """ ord() id to chr() chr """
        trans = []
        for text, n in zip(texts, texts_len):
            n = n.numpy().item()
            ids = text[:n]
            trans.append(''.join([chr(i) for i in ids]))
        return trans

    def compute_metrics(self,
                        utts,
                        audio,
                        audio_len,
                        texts,
                        texts_len,
                        fout=None):
        cfg = self.config.decoding
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        errors_func = error_rate.char_errors if cfg.error_rate_type == 'cer' else error_rate.word_errors
        error_rate_func = error_rate.cer if cfg.error_rate_type == 'cer' else error_rate.wer

        vocab_list = self.test_loader.collate_fn.vocab_list

        target_transcripts = self.ordid2token(texts, texts_len)
        result_transcripts = self.model.decode(
            audio,
            audio_len,
            vocab_list,
            decoding_method=cfg.decoding_method,
            lang_model_path=cfg.lang_model_path,
            beam_alpha=cfg.alpha,
            beam_beta=cfg.beta,
            beam_size=cfg.beam_size,
            cutoff_prob=cfg.cutoff_prob,
            cutoff_top_n=cfg.cutoff_top_n,
            num_processes=cfg.num_proc_bsearch)

        for utt, target, result in zip(utts, target_transcripts,
                                       result_transcripts):
            errors, len_ref = errors_func(target, result)
            errors_sum += errors
            len_refs += len_ref
            num_ins += 1
            if fout:
                fout.write(utt + " " + result + "\n")
            logger.info("\nTarget Transcription: %s\nOutput Transcription: %s" %
                        (target, result))
            logger.info("Current error rate [%s] = %f" %
                        (cfg.error_rate_type, error_rate_func(target, result)))

        return dict(
            errors_sum=errors_sum,
            len_refs=len_refs,
            num_ins=num_ins,
            error_rate=errors_sum / len_refs,
            error_rate_type=cfg.error_rate_type)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        logger.info(f"Test Total Examples: {len(self.test_loader.dataset)}")
        self.model.eval()
        cfg = self.config
        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        with open(self.args.result_file, 'w') as fout:
            for i, batch in enumerate(self.test_loader):
                utts, audio, audio_len, texts, texts_len = batch
                metrics = self.compute_metrics(utts, audio, audio_len, texts,
                                               texts_len, fout)
                errors_sum += metrics['errors_sum']
                len_refs += metrics['len_refs']
                num_ins += metrics['num_ins']
                error_rate_type = metrics['error_rate_type']
                logger.info("Error rate [%s] (%d/?) = %f" %
                            (error_rate_type, num_ins, errors_sum / len_refs))

        # logging
        msg = "Test: "
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += "Final error rate [%s] (%d/%d) = %f" % (
            error_rate_type, num_ins, num_ins, errors_sum / len_refs)
        logger.info(msg)

    def run_test(self):
        self.resume_or_scratch()
        try:
            self.test()
        except KeyboardInterrupt:
            exit(-1)

    def export(self):
        infer_model = DeepSpeech2InferModel.from_pretrained(
            self.test_loader, self.config, self.args.checkpoint_path)
        infer_model.eval()
        feat_dim = self.test_loader.collate_fn.feature_size
        static_model = paddle.jit.to_static(
            infer_model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None, feat_dim],
                    dtype='float32'),  # audio, [B,T,D]
                paddle.static.InputSpec(shape=[None],
                                        dtype='int64'),  # audio_length, [B]
            ])
        logger.info(f"Export code: {static_model.forward.code}")
        paddle.jit.save(static_model, self.args.export_path)

    def run_export(self):
        try:
            self.export()
        except KeyboardInterrupt:
            exit(-1)

    def setup(self):
        """Setup the experiment.
        """
        paddle.set_device(self.args.device)

        self.setup_output_dir()
        self.setup_checkpointer()

        self.setup_dataloader()
        self.setup_model()

        self.iteration = 0
        self.epoch = 0

    def setup_model(self):
        config = self.config
        model = DeepSpeech2Model(
            feat_size=self.test_loader.collate_fn.feature_size,
            dict_size=self.test_loader.collate_fn.vocab_size,
            num_conv_layers=config.model.num_conv_layers,
            num_rnn_layers=config.model.num_rnn_layers,
            rnn_size=config.model.rnn_layer_size,
            use_gru=config.model.use_gru,
            share_rnn_weights=config.model.share_rnn_weights)
        self.model = model
        logger.info("Setup model!")

    def setup_dataloader(self):
        config = self.config.clone()
        config.defrost()
        # return raw text

        config.data.manifest = config.data.test_manifest
        # filter test examples, will cause less examples, but no mismatch with training
        # and can use large batch size , save training time, so filter test egs now.
        # config.data.min_input_len = 0.0  # second
        # config.data.max_input_len = float('inf')  # second
        # config.data.min_output_len = 0.0  # tokens
        # config.data.max_output_len = float('inf')  # tokens
        # config.data.min_output_input_ratio = 0.00
        # config.data.max_output_input_ratio = float('inf')
        test_dataset = ManifestDataset.from_config(config)

        config.collator.keep_transcription_text = True
        config.collator.augmentation_config = ""
        # return text ord id
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.decoding.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=SpeechCollator.from_config(config))
        logger.info("Setup test Dataloader!")

    def setup_output_dir(self):
        """Create a directory used for output.
        """
        # output dir
        if self.args.output:
            output_dir = Path(self.args.output).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(
                self.args.checkpoint_path).expanduser().parent.parent
            output_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
