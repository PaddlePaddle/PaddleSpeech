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
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from paddle import distributed as dist
from paddle import inference
from paddle.io import DataLoader
from yacs.config import CfgNode

from deepspeech.io.collator import SpeechCollator
from deepspeech.io.dataset import ManifestDataset
from deepspeech.io.sampler import SortagradBatchSampler
from deepspeech.io.sampler import SortagradDistributedBatchSampler
from deepspeech.models.ds2 import DeepSpeech2InferModel
from deepspeech.models.ds2 import DeepSpeech2Model
from deepspeech.models.ds2_online import DeepSpeech2InferModelOnline
from deepspeech.models.ds2_online import DeepSpeech2ModelOnline
from deepspeech.training.gradclip import ClipGradByGlobalNormWithLog
from deepspeech.training.trainer import Trainer
from deepspeech.utils import error_rate
from deepspeech.utils import layer_tools
from deepspeech.utils import mp_tools
from deepspeech.utils.log import Autolog
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
        config = self.config.clone()
        config.defrost()
        config.model.feat_size = self.train_loader.collate_fn.feature_size
        config.model.dict_size = self.train_loader.collate_fn.vocab_size
        config.freeze()

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

        config.data.manifest = config.data.test_manifest
        test_dataset = ManifestDataset.from_config(config)

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

        config.collator.keep_transcription_text = True
        config.collator.augmentation_config = ""
        collate_fn_test = SpeechCollator.from_config(config)

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
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.decoding.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn_test)
        logger.info("Setup train/valid/test  Dataloader!")


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
        self.autolog.times.start()
        self.autolog.times.stamp()
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
        self.autolog.times.stamp()
        self.autolog.times.stamp()
        self.autolog.times.end()

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
        self.autolog = Autolog(
            batch_size=self.config.decoding.batch_size,
            model_name="deepspeech2",
            model_precision="fp32").getlog()
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
        self.autolog.report()

    def run_test(self):
        self.resume_or_scratch()
        try:
            self.test()
        except KeyboardInterrupt:
            exit(-1)

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


class DeepSpeech2ExportTester(DeepSpeech2Trainer):
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

        batch_size = self.config.decoding.batch_size

        output_prob_list = []
        output_lens_list = []
        decoder_chunk_size = 8
        subsampling_rate = self.model.encoder.conv.subsampling_rate
        receptive_field_length = self.model.encoder.conv.receptive_field_length
        chunk_stride = subsampling_rate * decoder_chunk_size
        chunk_size = (decoder_chunk_size - 1
                      ) * subsampling_rate + receptive_field_length

        x_batch = audio.numpy()
        x_len_batch = audio_len.numpy().astype(np.int64)
        max_len_batch = x_batch.shape[1]
        batch_padding_len = chunk_stride - (
            max_len_batch - chunk_size
        ) % chunk_stride  # The length of padding for the batch
        x_list = np.split(x_batch, x_batch.shape[0], axis=0)
        x_len_list = np.split(x_len_batch, x_batch.shape[0], axis=0)

        for x, x_len in zip(x_list, x_len_list):
            assert (chunk_size <= x_len[0])

            eouts_chunk_list = []
            eouts_chunk_lens_list = []

            padding_len_x = chunk_stride - (x_len[0] - chunk_size
                                            ) % chunk_stride
            padding = np.zeros(
                (x.shape[0], padding_len_x, x.shape[2]), dtype=np.float32)
            padded_x = np.concatenate([x, padding], axis=1)

            num_chunk = (x_len[0] + padding_len_x - chunk_size
                         ) / chunk_stride + 1
            num_chunk = int(num_chunk)

            chunk_state_h_box = np.zeros(
                (self.config.model.num_rnn_layers, 1,
                 self.config.model.rnn_layer_size),
                dtype=np.float32)
            chunk_state_c_box = np.zeros(
                (self.config.model.num_rnn_layers, 1,
                 self.config.model.rnn_layer_size),
                dtype=np.float32)

            input_names = self.predictor.get_input_names()
            audio_handle = self.predictor.get_input_handle(input_names[0])
            audio_len_handle = self.predictor.get_input_handle(input_names[1])
            h_box_handle = self.predictor.get_input_handle(input_names[2])
            c_box_handle = self.predictor.get_input_handle(input_names[3])

            probs_chunk_list = []
            probs_chunk_lens_list = []
            for i in range(0, num_chunk):
                start = i * chunk_stride
                end = start + chunk_size
                x_chunk = padded_x[:, start:end, :]
                x_len_left = np.where(x_len - i * chunk_stride < 0,
                                      np.zeros_like(x_len, dtype=np.int64),
                                      x_len - i * chunk_stride)
                x_chunk_len_tmp = np.ones_like(
                    x_len, dtype=np.int64) * chunk_size
                x_chunk_lens = np.where(x_len_left < x_chunk_len_tmp,
                                        x_len_left, x_chunk_len_tmp)
                if (x_chunk_lens[0] <
                        receptive_field_length):  #means the number of input frames in the chunk is not enough for predicting one prob
                    break
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
                output_chunk_prob = output_handle.copy_to_cpu()
                output_chunk_lens = output_lens_handle.copy_to_cpu()
                chunk_state_h_box = output_state_h_handle.copy_to_cpu()
                chunk_state_c_box = output_state_c_handle.copy_to_cpu()
                output_chunk_prob = paddle.to_tensor(output_chunk_prob)
                output_chunk_lens = paddle.to_tensor(output_chunk_lens)

                probs_chunk_list.append(output_chunk_prob)
                probs_chunk_lens_list.append(output_chunk_lens)
            output_prob = paddle.concat(probs_chunk_list, axis=1)
            output_lens = paddle.add_n(probs_chunk_lens_list)
            output_prob_padding_len = max_len_batch + batch_padding_len - output_prob.shape[
                1]
            output_prob_padding = paddle.zeros(
                (1, output_prob_padding_len, output_prob.shape[2]),
                dtype="float32")  # The prob padding for a piece of utterance
            output_prob = paddle.concat(
                [output_prob, output_prob_padding], axis=1)
            output_prob_list.append(output_prob)
            output_lens_list.append(output_lens)
        output_prob_branch = paddle.concat(output_prob_list, axis=0)
        output_lens_branch = paddle.concat(output_lens_list, axis=0)
        """
        x = audio.numpy()
        x_len = audio_len.numpy().astype(np.int64)

        input_names = self.predictor.get_input_names()
        audio_handle = self.predictor.get_input_handle(input_names[0])
        audio_len_handle = self.predictor.get_input_handle(input_names[1])
        h_box_handle = self.predictor.get_input_handle(input_names[2])
        c_box_handle = self.predictor.get_input_handle(input_names[3])


        audio_handle.reshape(x.shape)
        audio_handle.copy_from_cpu(x)

        audio_len_handle.reshape(x_len.shape)
        audio_len_handle.copy_from_cpu(x_len)

        init_state_h_box = np.zeros((self.config.model.num_rnn_layers, audio.shape[0], self.config.model.rnn_layer_size), dtype=np.float32)
        init_state_c_box = np.zeros((self.config.model.num_rnn_layers, audio.shape[0], self.config.model.rnn_layer_size), dtype=np.float32)
        h_box_handle.reshape(init_state_h_box.shape)
        h_box_handle.copy_from_cpu(init_state_h_box)

        c_box_handle.reshape(init_state_c_box.shape)
        c_box_handle.copy_from_cpu(init_state_c_box)

        #self.autolog.times.start()
        #self.autolog.times.stamp()
        self.predictor.run()

        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        output_lens_handle = self.predictor.get_output_handle(output_names[1])
        output_state_h_handle = self.predictor.get_output_handle(output_names[2])
        output_state_c_handle = self.predictor.get_output_handle(output_names[3])
        output_prob = output_handle.copy_to_cpu()
        output_lens = output_lens_handle.copy_to_cpu()
        output_stata_h_box = output_state_h_handle.copy_to_cpu()
        output_stata_c_box = output_state_c_handle.copy_to_cpu()
        output_prob_branch = paddle.to_tensor(output_prob)
        output_lens_branch = paddle.to_tensor(output_lens)
        """

        result_transcripts = self.model.decode_by_probs(
            output_prob_branch,
            output_lens_branch,
            vocab_list,
            decoding_method=cfg.decoding_method,
            lang_model_path=cfg.lang_model_path,
            beam_alpha=cfg.alpha,
            beam_beta=cfg.beta,
            beam_size=cfg.beam_size,
            cutoff_prob=cfg.cutoff_prob,
            cutoff_top_n=cfg.cutoff_top_n,
            num_processes=cfg.num_proc_bsearch)

        #self.autolog.times.stamp()
        #self.autolog.times.stamp()
        #self.autolog.times.end()
        target_transcripts = self.ordid2token(texts, texts_len)
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
        #self.autolog = Autolog(
        #    batch_size=self.config.decoding.batch_size,
        #    model_name="deepspeech2",
        #    model_precision="fp32").getlog()
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
        #self.autolog.report()

    def run_test(self):
        try:
            self.test()
        except KeyboardInterrupt:
            exit(-1)

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
        #self.setup_checkpointer()

        self.setup_dataloader()
        self.setup_model()

        self.iteration = 0
        self.epoch = 0

    def setup_output_dir(self):
        """Create a directory used for output.
        """
        # output dir
        if self.args.output:
            output_dir = Path(self.args.output).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(self.args.export_path).expanduser().parent.parent
            output_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir

    def setup_model(self):
        super().setup_model()
        if self.args.model_type == 'online':
            #inference_dir = "exp/deepspeech2_online/checkpoints/"
            #inference_dir = "exp/deepspeech2_online_3rr_1fc_lr_decay0.91_lstm/checkpoints/"
            #speedyspeech_config = inference.Config(
            #    str(Path(inference_dir) / "avg_1.jit.pdmodel"),
            #    str(Path(inference_dir) / "avg_1.jit.pdiparams"))
            speedyspeech_config = inference.Config(
                self.args.export_path + ".pdmodel",
                self.args.export_path + ".pdiparams")
            speedyspeech_config.enable_use_gpu(100, 0)
            speedyspeech_config.enable_memory_optim()
            speedyspeech_predictor = inference.create_predictor(
                speedyspeech_config)
            self.predictor = speedyspeech_predictor
