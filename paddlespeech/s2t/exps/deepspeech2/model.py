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
"""Contains DeepSpeech2 and DeepSpeech2Online model."""
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Optional

import jsonlines
import numpy as np
import paddle
from paddle import distributed as dist
from paddle import inference
from paddle.io import DataLoader
from yacs.config import CfgNode

from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.io.dataset import ManifestDataset
from paddlespeech.s2t.io.sampler import SortagradBatchSampler
from paddlespeech.s2t.io.sampler import SortagradDistributedBatchSampler
from paddlespeech.s2t.models.ds2 import DeepSpeech2InferModel
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model
from paddlespeech.s2t.models.ds2_online import DeepSpeech2InferModelOnline
from paddlespeech.s2t.models.ds2_online import DeepSpeech2ModelOnline
from paddlespeech.s2t.training.gradclip import ClipGradByGlobalNormWithLog
from paddlespeech.s2t.training.reporter import report
from paddlespeech.s2t.training.timer import Timer
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils import error_rate
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig

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
        batch_size = self.config.collator.batch_size
        accum_grad = self.config.training.accum_grad

        start = time.time()

        # forward
        utt, audio, audio_len, text, text_len = batch_data
        loss = self.model(audio, audio_len, text, text_len)
        losses_np = {
            'train_loss': float(loss),
        }

        # loss backward
        if (batch_index + 1) % accum_grad != 0:
            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            context = self.model.no_sync if (hasattr(self.model, "no_sync") and
                                             self.parallel) else nullcontext
        else:
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            context = nullcontext

        with context():
            loss.backward()
            layer_tools.print_grads(self.model, print_func=None)

        # optimizer step
        if (batch_index + 1) % accum_grad == 0:
            self.optimizer.step()
            self.optimizer.clear_grad()
            self.iteration += 1

        iteration_time = time.time() - start

        for k, v in losses_np.items():
            report(k, v)
        report("batch_size", batch_size)
        report("accum", accum_grad)
        report("step_cost", iteration_time)

        if dist.get_rank() == 0 and self.visualizer:
            for k, v in losses_np.items():
                # `step -1` since we update `step` after optimizer.step().
                self.visualizer.add_scalar("train/{}".format(k), v,
                                           self.iteration - 1)

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
        config = self.config.clone()
        with UpdateConfig(config):
            if self.train:
                config.model.input_dim = self.train_loader.collate_fn.feature_size
                config.model.output_dim = self.train_loader.collate_fn.vocab_size
            else:
                config.model.input_dim = self.test_loader.collate_fn.feature_size
                config.model.output_dim = self.test_loader.collate_fn.vocab_size

        if self.args.model_type == 'offline':
            model = DeepSpeech2Model.from_config(config.model)
        elif self.args.model_type == 'online':
            model = DeepSpeech2ModelOnline.from_config(config.model)
        else:
            raise Exception("wrong model type")
        if self.parallel:
            model = paddle.DataParallel(model)

        logger.info(f"{model}")
        layer_tools.print_params(model, logger.info)
        self.model = model
        logger.info("Setup model!")

        if not self.train:
            return

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
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        logger.info("Setup optimizer/lr_scheduler!")

    def setup_dataloader(self):
        config = self.config.clone()
        config.defrost()
        if self.train:
            # train
            config.data.manifest = config.data.train_manifest
            train_dataset = ManifestDataset.from_config(config)
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

            config.collator.keep_transcription_text = False
            collate_fn_train = SpeechCollator.from_config(config)
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn_train,
                num_workers=config.collator.num_workers)

            # dev
            config.data.manifest = config.data.dev_manifest
            dev_dataset = ManifestDataset.from_config(config)

            config.collator.augmentation_config = ""
            config.collator.keep_transcription_text = False
            collate_fn_dev = SpeechCollator.from_config(config)
            self.valid_loader = DataLoader(
                dev_dataset,
                batch_size=int(config.collator.batch_size),
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn_dev,
                num_workers=config.collator.num_workers)
            logger.info("Setup train/valid  Dataloader!")
        else:
            # test
            config.data.manifest = config.data.test_manifest
            test_dataset = ManifestDataset.from_config(config)

            config.collator.augmentation_config = ""
            config.collator.keep_transcription_text = True
            collate_fn_test = SpeechCollator.from_config(config)

            self.test_loader = DataLoader(
                test_dataset,
                batch_size=config.decoding.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn_test,
                num_workers=config.collator.num_workers)
            logger.info("Setup test  Dataloader!")


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
        self._text_featurizer = TextFeaturizer(
            unit_type=config.collator.unit_type, vocab_filepath=None)

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

        result_transcripts = self.compute_result_transcripts(audio, audio_len,
                                                             vocab_list, cfg)

        for utt, target, result in zip(utts, target_transcripts,
                                       result_transcripts):
            errors, len_ref = errors_func(target, result)
            errors_sum += errors
            len_refs += len_ref
            num_ins += 1
            if fout:
                fout.write({"utt": utt, "ref": target, "hyp": result})
            logger.info(f"Utt: {utt}")
            logger.info(f"Ref: {target}")
            logger.info(f"Hyp: {result}")
            logger.info("Current error rate [%s] = %f" %
                        (cfg.error_rate_type, error_rate_func(target, result)))

        return dict(
            errors_sum=errors_sum,
            len_refs=len_refs,
            num_ins=num_ins,
            error_rate=errors_sum / len_refs,
            error_rate_type=cfg.error_rate_type)

    def compute_result_transcripts(self, audio, audio_len, vocab_list, cfg):
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

        return result_transcripts

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        logger.info(f"Test Total Examples: {len(self.test_loader.dataset)}")
        self.model.eval()
        cfg = self.config
        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        with jsonlines.open(self.args.result_file, 'w') as fout:
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

    @paddle.no_grad()
    def export(self):
        if self.args.model_type == 'offline':
            infer_model = DeepSpeech2InferModel.from_pretrained(
                self.test_loader, self.config, self.args.checkpoint_path)
        elif self.args.model_type == 'online':
            infer_model = DeepSpeech2InferModelOnline.from_pretrained(
                self.test_loader, self.config, self.args.checkpoint_path)
        else:
            raise Exception("wrong model type")

        infer_model.eval()
        feat_dim = self.test_loader.collate_fn.feature_size
        static_model = infer_model.export()
        logger.info(f"Export code: {static_model.forward.code}")
        paddle.jit.save(static_model, self.args.export_path)


class DeepSpeech2ExportTester(DeepSpeech2Tester):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.apply_static = True
        self.args = args

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        logger.info(f"Test Total Examples: {len(self.test_loader.dataset)}")
        if self.args.enable_auto_log is True:
            from paddlespeech.s2t.utils.log import Autolog
            self.autolog = Autolog(
                batch_size=self.config.decoding.batch_size,
                model_name="deepspeech2",
                model_precision="fp32").getlog()
        self.model.eval()
        cfg = self.config
        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        with jsonlines.open(self.args.result_file, 'w') as fout:
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
        if self.args.enable_auto_log is True:
            self.autolog.report()

    def compute_result_transcripts(self, audio, audio_len, vocab_list, cfg):
        if self.args.model_type == "online":
            output_probs, output_lens = self.static_forward_online(audio,
                                                                   audio_len)
        elif self.args.model_type == "offline":
            output_probs, output_lens = self.static_forward_offline(audio,
                                                                    audio_len)
        else:
            raise Exception("wrong model type")

        self.predictor.clear_intermediate_tensor()
        self.predictor.try_shrink_memory()

        self.model.decoder.init_decode(cfg.alpha, cfg.beta, cfg.lang_model_path,
                                       vocab_list, cfg.decoding_method)

        result_transcripts = self.model.decoder.decode_probs(
            output_probs, output_lens, vocab_list, cfg.decoding_method,
            cfg.lang_model_path, cfg.alpha, cfg.beta, cfg.beam_size,
            cfg.cutoff_prob, cfg.cutoff_top_n, cfg.num_proc_bsearch)
        #replace the <space> with ' '
        result_transcripts = [
            self._text_featurizer.detokenize(sentence)
            for sentence in result_transcripts
        ]

        return result_transcripts

    def run_test(self):
        """Do Test/Decode"""
        try:
            with Timer("Test/Decode Done: {}"):
                with self.eval():
                    self.test()
        except KeyboardInterrupt:
            exit(-1)

    def static_forward_online(self, audio, audio_len,
                              decoder_chunk_size: int=1):
        """
        Parameters
        ----------
            audio (Tensor): shape[B, T, D]
            audio_len (Tensor): shape[B]
            decoder_chunk_size(int)
        Returns
        -------
            output_probs(numpy.array): shape[B, T, vocab_size]
            output_lens(numpy.array): shape[B]
        """
        output_probs_list = []
        output_lens_list = []
        subsampling_rate = self.model.encoder.conv.subsampling_rate
        receptive_field_length = self.model.encoder.conv.receptive_field_length
        chunk_stride = subsampling_rate * decoder_chunk_size
        chunk_size = (decoder_chunk_size - 1
                      ) * subsampling_rate + receptive_field_length

        x_batch = audio.numpy()
        batch_size, Tmax, x_dim = x_batch.shape
        x_len_batch = audio_len.numpy().astype(np.int64)
        if (Tmax - chunk_size) % chunk_stride != 0:
            padding_len_batch = chunk_stride - (
                Tmax - chunk_size
            ) % chunk_stride  # The length of padding for the batch
        else:
            padding_len_batch = 0
        x_list = np.split(x_batch, batch_size, axis=0)
        x_len_list = np.split(x_len_batch, batch_size, axis=0)

        for x, x_len in zip(x_list, x_len_list):
            if self.args.enable_auto_log is True:
                self.autolog.times.start()
            x_len = x_len[0]
            assert (chunk_size <= x_len)

            if (x_len - chunk_size) % chunk_stride != 0:
                padding_len_x = chunk_stride - (x_len - chunk_size
                                                ) % chunk_stride
            else:
                padding_len_x = 0

            padding = np.zeros(
                (x.shape[0], padding_len_x, x.shape[2]), dtype=x.dtype)
            padded_x = np.concatenate([x, padding], axis=1)

            num_chunk = (x_len + padding_len_x - chunk_size) / chunk_stride + 1
            num_chunk = int(num_chunk)

            chunk_state_h_box = np.zeros(
                (self.config.model.num_rnn_layers, 1,
                 self.config.model.rnn_layer_size),
                dtype=x.dtype)
            chunk_state_c_box = np.zeros(
                (self.config.model.num_rnn_layers, 1,
                 self.config.model.rnn_layer_size),
                dtype=x.dtype)

            input_names = self.predictor.get_input_names()
            audio_handle = self.predictor.get_input_handle(input_names[0])
            audio_len_handle = self.predictor.get_input_handle(input_names[1])
            h_box_handle = self.predictor.get_input_handle(input_names[2])
            c_box_handle = self.predictor.get_input_handle(input_names[3])

            probs_chunk_list = []
            probs_chunk_lens_list = []
            if self.args.enable_auto_log is True:
                # record the model preprocessing time
                self.autolog.times.stamp()

            for i in range(0, num_chunk):
                start = i * chunk_stride
                end = start + chunk_size
                x_chunk = padded_x[:, start:end, :]
                if x_len < i * chunk_stride:
                    x_chunk_lens = 0
                else:
                    x_chunk_lens = min(x_len - i * chunk_stride, chunk_size)

                if (x_chunk_lens <
                        receptive_field_length):  #means the number of input frames in the chunk is not enough for predicting one prob
                    break
                x_chunk_lens = np.array([x_chunk_lens])
                audio_handle.reshape(x_chunk.shape)
                audio_handle.copy_from_cpu(x_chunk)

                audio_len_handle.reshape(x_chunk_lens.shape)
                audio_len_handle.copy_from_cpu(x_chunk_lens)

                h_box_handle.reshape(chunk_state_h_box.shape)
                h_box_handle.copy_from_cpu(chunk_state_h_box)

                c_box_handle.reshape(chunk_state_c_box.shape)
                c_box_handle.copy_from_cpu(chunk_state_c_box)

                output_names = self.predictor.get_output_names()
                output_handle = self.predictor.get_output_handle(
                    output_names[0])
                output_lens_handle = self.predictor.get_output_handle(
                    output_names[1])
                output_state_h_handle = self.predictor.get_output_handle(
                    output_names[2])
                output_state_c_handle = self.predictor.get_output_handle(
                    output_names[3])
                self.predictor.run()
                output_chunk_probs = output_handle.copy_to_cpu()
                output_chunk_lens = output_lens_handle.copy_to_cpu()
                chunk_state_h_box = output_state_h_handle.copy_to_cpu()
                chunk_state_c_box = output_state_c_handle.copy_to_cpu()

                probs_chunk_list.append(output_chunk_probs)
                probs_chunk_lens_list.append(output_chunk_lens)
            output_probs = np.concatenate(probs_chunk_list, axis=1)
            output_lens = np.sum(probs_chunk_lens_list, axis=0)
            vocab_size = output_probs.shape[2]
            output_probs_padding_len = Tmax + padding_len_batch - output_probs.shape[
                1]
            output_probs_padding = np.zeros(
                (1, output_probs_padding_len, vocab_size),
                dtype=output_probs.
                dtype)  # The prob padding for a piece of utterance
            output_probs = np.concatenate(
                [output_probs, output_probs_padding], axis=1)
            output_probs_list.append(output_probs)
            output_lens_list.append(output_lens)
            if self.args.enable_auto_log is True:
                # record the model inference time
                self.autolog.times.stamp()
                # record the post processing time
                self.autolog.times.stamp()
                self.autolog.times.end()
        output_probs = np.concatenate(output_probs_list, axis=0)
        output_lens = np.concatenate(output_lens_list, axis=0)
        return output_probs, output_lens

    def static_forward_offline(self, audio, audio_len):
        """
        Parameters
        ----------
            audio (Tensor): shape[B, T, D]
            audio_len (Tensor): shape[B]

        Returns
        -------
            output_probs(numpy.array): shape[B, T, vocab_size]
            output_lens(numpy.array): shape[B]
        """
        x = audio.numpy()
        x_len = audio_len.numpy().astype(np.int64)

        input_names = self.predictor.get_input_names()
        audio_handle = self.predictor.get_input_handle(input_names[0])
        audio_len_handle = self.predictor.get_input_handle(input_names[1])

        audio_handle.reshape(x.shape)
        audio_handle.copy_from_cpu(x)

        audio_len_handle.reshape(x_len.shape)
        audio_len_handle.copy_from_cpu(x_len)

        if self.args.enable_auto_log is True:
            self.autolog.times.start()
            # record the prefix processing time
            self.autolog.times.stamp()
        self.predictor.run()
        if self.args.enable_auto_log is True:
            # record the model inference time
            self.autolog.times.stamp()
            # record the post processing time
            self.autolog.times.stamp()
            self.autolog.times.end()

        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        output_lens_handle = self.predictor.get_output_handle(output_names[1])
        output_probs = output_handle.copy_to_cpu()
        output_lens = output_lens_handle.copy_to_cpu()
        return output_probs, output_lens

    def setup_model(self):
        super().setup_model()
        deepspeech_config = inference.Config(
            self.args.export_path + ".pdmodel",
            self.args.export_path + ".pdiparams")
        if (os.environ['CUDA_VISIBLE_DEVICES'].strip() != ''):
            deepspeech_config.enable_use_gpu(100, 0)
            deepspeech_config.enable_memory_optim()
        deepspeech_predictor = inference.create_predictor(deepspeech_config)
        self.predictor = deepspeech_predictor
