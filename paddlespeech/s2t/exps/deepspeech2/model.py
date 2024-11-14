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

import jsonlines
import numpy as np
import paddle
from paddle import distributed as dist
from paddle import inference

from paddlespeech.audio.text.text_featurizer import TextFeaturizer
from paddlespeech.s2t.io.dataloader import BatchDataLoader
from paddlespeech.s2t.models.ds2 import DeepSpeech2InferModel
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model
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
    def __init__(self, config, args):
        super().__init__(config, args)

    def train_batch(self, batch_index, batch_data, msg):
        batch_size = self.config.batch_size
        accum_grad = self.config.accum_grad

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

            if (i + 1) % self.config.log_interval == 0:
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
                config.input_dim = self.train_loader.feat_dim
                config.output_dim = self.train_loader.vocab_size
            else:
                config.input_dim = self.test_loader.feat_dim
                config.output_dim = self.test_loader.vocab_size

        model = DeepSpeech2Model.from_config(config)
        if self.parallel:
            model = paddle.DataParallel(model)

        logger.info(f"{model}")
        layer_tools.print_params(model, logger.info)
        self.model = model
        logger.info("Setup model!")

        if not self.train:
            return

        grad_clip = paddle.nn.ClipGradByGlobalNorm(config.global_grad_clip)
        lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=config.lr, gamma=config.lr_decay, verbose=True)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=paddle.regularizer.L2Decay(config.weight_decay),
            grad_clip=grad_clip)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        logger.info("Setup optimizer/lr_scheduler!")

    def setup_dataloader(self):
        config = self.config.clone()
        config.defrost()
        if self.train:
            # train/valid dataset, return token ids
            self.train_loader = BatchDataLoader(
                json_file=config.train_manifest,
                train_mode=True,
                sortagrad=config.sortagrad,
                batch_size=config.batch_size,
                maxlen_in=config.maxlen_in,
                maxlen_out=config.maxlen_out,
                minibatches=config.minibatches,
                mini_batch_size=self.args.ngpu,
                batch_count=config.batch_count,
                batch_bins=config.batch_bins,
                batch_frames_in=config.batch_frames_in,
                batch_frames_out=config.batch_frames_out,
                batch_frames_inout=config.batch_frames_inout,
                preprocess_conf=config.preprocess_config,
                n_iter_processes=config.num_workers,
                subsampling_factor=1,
                num_encs=1,
                dist_sampler=config.get('dist_sampler', False),
                shortest_first=False)

            self.valid_loader = BatchDataLoader(
                json_file=config.dev_manifest,
                train_mode=False,
                sortagrad=False,
                batch_size=config.batch_size,
                maxlen_in=float('inf'),
                maxlen_out=float('inf'),
                minibatches=0,
                mini_batch_size=self.args.ngpu,
                batch_count='auto',
                batch_bins=0,
                batch_frames_in=0,
                batch_frames_out=0,
                batch_frames_inout=0,
                preprocess_conf=config.preprocess_config,
                n_iter_processes=config.num_workers,
                subsampling_factor=1,
                num_encs=1,
                dist_sampler=config.get('dist_sampler', False),
                shortest_first=False)
            logger.info("Setup train/valid Dataloader!")
        else:
            decode_batch_size = config.get('decode', dict()).get(
                'decode_batch_size', 1)
            # test dataset, return raw text
            self.test_loader = BatchDataLoader(
                json_file=config.test_manifest,
                train_mode=False,
                sortagrad=False,
                batch_size=decode_batch_size,
                maxlen_in=float('inf'),
                maxlen_out=float('inf'),
                minibatches=0,
                mini_batch_size=1,
                batch_count='auto',
                batch_bins=0,
                batch_frames_in=0,
                batch_frames_out=0,
                batch_frames_inout=0,
                preprocess_conf=config.preprocess_config,
                n_iter_processes=1,
                subsampling_factor=1,
                num_encs=1)
            logger.info("Setup test/align Dataloader!")


class DeepSpeech2Tester(DeepSpeech2Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)
        self._text_featurizer = TextFeaturizer(
            unit_type=config.unit_type, vocab=config.vocab_filepath)
        self.vocab_list = self._text_featurizer.vocab_list

    def ordid2token(self, texts, texts_len):
        """ ord() id to chr() chr """
        trans = []
        for text, n in zip(texts, texts_len):
            n = n.numpy().item()
            ids = text[:n]
            trans.append(
                self._text_featurizer.defeaturize(ids.numpy().tolist()))
        return trans

    def compute_metrics(self,
                        utts,
                        audio,
                        audio_len,
                        texts,
                        texts_len,
                        fout=None):
        decode_cfg = self.config.decode
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        errors_func = error_rate.char_errors if decode_cfg.error_rate_type == 'cer' else error_rate.word_errors
        error_rate_func = error_rate.cer if decode_cfg.error_rate_type == 'cer' else error_rate.wer

        target_transcripts = self.ordid2token(texts, texts_len)

        result_transcripts = self.compute_result_transcripts(audio, audio_len)

        for utt, target, result in zip(utts, target_transcripts,
                                       result_transcripts):
            errors, len_ref = errors_func(target, result)
            errors_sum += errors
            len_refs += len_ref
            num_ins += 1
            if fout:
                fout.write({"utt": utt, "refs": [target], "hyps": [result]})
            logger.info(f"Utt: {utt}")
            logger.info(f"Ref: {target}")
            logger.info(f"Hyp: {result}")
            logger.info(
                "Current error rate [%s] = %f" %
                (decode_cfg.error_rate_type, error_rate_func(target, result)))

        return dict(
            errors_sum=errors_sum,
            len_refs=len_refs,
            num_ins=num_ins,
            error_rate=errors_sum / len_refs,
            error_rate_type=decode_cfg.error_rate_type)

    def compute_result_transcripts(self, audio, audio_len):
        result_transcripts = self.model.decode(audio, audio_len)
        return result_transcripts

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        logger.info(f"Test Total Examples: {len(self.test_loader.dataset)}")
        self.model.eval()
        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0

        # Initialized the decoder in model
        decode_cfg = self.config.decode
        vocab_list = self.vocab_list
        decode_batch_size = decode_cfg.decode_batch_size
        self.model.decoder.init_decoder(
            decode_batch_size, vocab_list, decode_cfg.decoding_method,
            decode_cfg.lang_model_path, decode_cfg.alpha, decode_cfg.beta,
            decode_cfg.beam_size, decode_cfg.cutoff_prob,
            decode_cfg.cutoff_top_n, decode_cfg.num_proc_bsearch)

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
        self.model.decoder.del_decoder()

    @paddle.no_grad()
    def export(self):
        infer_model = DeepSpeech2InferModel.from_pretrained(
            self.test_loader, self.config, self.args.checkpoint_path)
        infer_model.eval()
        static_model = infer_model.export()
        try:
            logger.info(f"Export code: {static_model.forward.code}")
        except:
            logger.info(
                f"Fail to print Export code, static_model.forward.code can not be run."
            )
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
                batch_size=self.config.decode.decode_batch_size,
                model_name="deepspeech2",
                model_precision="fp32").getlog()
        self.model.eval()
        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0

        # Initialized the decoder in model
        decode_cfg = self.config.decode
        vocab_list = self.vocab_list
        if self.config.rnn_direction == "forward":
            decode_batch_size = 1
        elif self.config.rnn_direction == "bidirect":
            decode_batch_size = self.test_loader.batch_size
        else:
            raise Exception("wrong model type")
        self.model.decoder.init_decoder(
            decode_batch_size, vocab_list, decode_cfg.decoding_method,
            decode_cfg.lang_model_path, decode_cfg.alpha, decode_cfg.beta,
            decode_cfg.beam_size, decode_cfg.cutoff_prob,
            decode_cfg.cutoff_top_n, decode_cfg.num_proc_bsearch)

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
        self.model.decoder.del_decoder()

    def compute_result_transcripts(self, audio, audio_len):
        if self.config.rnn_direction == "forward":
            output_probs, output_lens, trans_batch = self.static_forward_online(
                audio, audio_len, decoder_chunk_size=1)
            result_transcripts = [trans[-1] for trans in trans_batch]
        elif self.config.rnn_direction == "bidirect":
            output_probs, output_lens = self.static_forward_offline(audio,
                                                                    audio_len)
            batch_size = output_probs.shape[0]
            self.model.decoder.reset_decoder(batch_size=batch_size)

            self.model.decoder.next(output_probs, output_lens)

            trans_best, trans_beam = self.model.decoder.decode()

            result_transcripts = trans_best

        else:
            raise Exception("wrong model type")

        self.predictor.clear_intermediate_tensor()
        self.predictor.try_shrink_memory()

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
            trans(list(list(str))): shape[B, T]
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
            # The length of padding for the batch
            padding_len_batch = chunk_stride - (Tmax - chunk_size
                                                ) % chunk_stride
        else:
            padding_len_batch = 0
        x_list = np.split(x_batch, batch_size, axis=0)
        x_len_list = np.split(x_len_batch, batch_size, axis=0)

        trans_batch = []
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
                (self.config.num_rnn_layers, 1, self.config.rnn_layer_size),
                dtype=x.dtype)
            chunk_state_c_box = np.zeros(
                (self.config.num_rnn_layers, 1, self.config.rnn_layer_size),
                dtype=x.dtype)

            input_names = self.predictor.get_input_names()
            audio_handle = self.predictor.get_input_handle(input_names[0])
            audio_len_handle = self.predictor.get_input_handle(input_names[1])
            h_box_handle = self.predictor.get_input_handle(input_names[2])
            c_box_handle = self.predictor.get_input_handle(input_names[3])

            trans = []
            probs_chunk_list = []
            probs_chunk_lens_list = []
            if self.args.enable_auto_log is True:
                # record the model preprocessing time
                self.autolog.times.stamp()

            self.model.decoder.reset_decoder(batch_size=1)
            for i in range(0, num_chunk):
                start = i * chunk_stride
                end = start + chunk_size
                x_chunk = padded_x[:, start:end, :]
                if x_len < i * chunk_stride:
                    x_chunk_lens = 0
                else:
                    x_chunk_lens = min(x_len - i * chunk_stride, chunk_size)
                #means the number of input frames in the chunk is not enough for predicting one prob
                if (x_chunk_lens < receptive_field_length):
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
                self.model.decoder.next(output_chunk_probs, output_chunk_lens)
                probs_chunk_list.append(output_chunk_probs)
                probs_chunk_lens_list.append(output_chunk_lens)
                trans_best, trans_beam = self.model.decoder.decode()
                trans.append(trans_best[0])
            trans_batch.append(trans)
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
        return output_probs, output_lens, trans_batch

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
