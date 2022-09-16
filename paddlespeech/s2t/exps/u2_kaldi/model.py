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
"""Contains U2 model."""
import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext

import jsonlines
import numpy as np
import paddle
from paddle import distributed as dist

from paddlespeech.s2t.frontend.featurizer import TextFeaturizer
from paddlespeech.s2t.frontend.utility import load_dict
from paddlespeech.s2t.io.dataloader import DataLoaderFactory
from paddlespeech.s2t.models.u2 import U2Model
from paddlespeech.s2t.training.optimizer import OptimizerFactory
from paddlespeech.s2t.training.scheduler import LRSchedulerFactory
from paddlespeech.s2t.training.timer import Timer
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils import ctc_utils
from paddlespeech.s2t.utils import error_rate
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig

logger = Log(__name__).getlog()


class U2Trainer(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)

    def train_batch(self, batch_index, batch_data, msg):
        train_conf = self.config
        start = time.time()

        # forward
        utt, audio, audio_len, text, text_len = batch_data
        loss, attention_loss, ctc_loss = self.model(audio, audio_len, text,
                                                    text_len)

        # loss div by `batch_size * accum_grad`
        loss /= train_conf.accum_grad
        losses_np = {'loss': float(loss) * train_conf.accum_grad}
        if attention_loss:
            losses_np['att_loss'] = float(attention_loss)
        if ctc_loss:
            losses_np['ctc_loss'] = float(ctc_loss)

        # loss backward
        if (batch_index + 1) % train_conf.accum_grad != 0:
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
        if (batch_index + 1) % train_conf.accum_grad == 0:
            self.optimizer.step()
            self.optimizer.clear_grad()
            self.lr_scheduler.step()
            self.iteration += 1

        iteration_time = time.time() - start

        if (batch_index + 1) % train_conf.log_interval == 0:
            msg += "train time: {:>.3f}s, ".format(iteration_time)
            msg += "batch size: {}, ".format(self.config.batch_size)
            msg += "accum: {}, ".format(train_conf.accum_grad)
            msg += ', '.join('{}: {:>.6f}'.format(k, v)
                             for k, v in losses_np.items())
            logger.info(msg)

            if dist.get_rank() == 0 and self.visualizer:
                losses_np_v = losses_np.copy()
                losses_np_v.update({"lr": self.lr_scheduler()})
                for key, val in losses_np_v.items():
                    self.visualizer.add_scalar(
                        tag="train/" + key, value=val, step=self.iteration - 1)

    @paddle.no_grad()
    def valid(self):
        self.model.eval()
        if not self.use_streamdata:
            logger.info(
                f"Valid Total Examples: {len(self.valid_loader.dataset)}")
        valid_losses = defaultdict(list)
        num_seen_utts = 1
        total_loss = 0.0

        for i, batch in enumerate(self.valid_loader):
            utt, audio, audio_len, text, text_len = batch
            loss, attention_loss, ctc_loss = self.model(audio, audio_len, text,
                                                        text_len)
            if paddle.isfinite(loss):
                num_utts = batch[1].shape[0]
                num_seen_utts += num_utts
                total_loss += float(loss) * num_utts
                valid_losses['val_loss'].append(float(loss))
                if attention_loss:
                    valid_losses['val_att_loss'].append(float(attention_loss))
                if ctc_loss:
                    valid_losses['val_ctc_loss'].append(float(ctc_loss))

            if (i + 1) % self.config.log_interval == 0:
                valid_dump = {k: np.mean(v) for k, v in valid_losses.items()}
                valid_dump['val_history_loss'] = total_loss / num_seen_utts

                # logging
                msg = f"Valid: Rank: {dist.get_rank()}, "
                msg += "epoch: {}, ".format(self.epoch)
                msg += "step: {}, ".format(self.iteration)
                if not self.use_streamdata:
                    msg += "batch: {}/{}, ".format(i + 1,
                                                   len(self.valid_loader))
                msg += ', '.join('{}: {:>.6f}'.format(k, v)
                                 for k, v in valid_dump.items())
                logger.info(msg)

        logger.info('Rank {} Val info val_loss {}'.format(
            dist.get_rank(), total_loss / num_seen_utts))
        return total_loss, num_seen_utts

    def do_train(self):
        """The training process control by step."""
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        # script_model = paddle.jit.to_static(self.model)
        # script_model_path = str(self.checkpoint_dir / 'init')
        # paddle.jit.save(script_model, script_model_path)

        self.before_train()
        if not self.use_streamdata:
            logger.info(
                f"Train Total Examples: {len(self.train_loader.dataset)}")
        while self.epoch < self.config.n_epoch:
            with Timer("Epoch-Train Time Cost: {}"):
                self.model.train()
                try:
                    data_start_time = time.time()
                    for batch_index, batch in enumerate(self.train_loader):
                        dataload_time = time.time() - data_start_time
                        msg = "Train: Rank: {}, ".format(dist.get_rank())
                        msg += "epoch: {}, ".format(self.epoch)
                        msg += "step: {}, ".format(self.iteration)
                        if not self.use_streamdata:
                            msg += "batch : {}/{}, ".format(
                                batch_index + 1, len(self.train_loader))
                        msg += "lr: {:>.8f}, ".format(self.lr_scheduler())
                        msg += "data time: {:>.3f}s, ".format(dataload_time)
                        self.train_batch(batch_index, batch, msg)
                        self.after_train_batch()
                        data_start_time = time.time()
                except Exception as e:
                    logger.error(e)
                    raise e

            with Timer("Eval Time Cost: {}"):
                total_loss, num_seen_utts = self.valid()
                if dist.get_world_size() > 1:
                    num_seen_utts = paddle.to_tensor(num_seen_utts)
                    # the default operator in all_reduce function is sum.
                    dist.all_reduce(num_seen_utts)
                    total_loss = paddle.to_tensor(total_loss)
                    dist.all_reduce(total_loss)
                    cv_loss = total_loss / num_seen_utts
                    cv_loss = float(cv_loss)
                else:
                    cv_loss = total_loss / num_seen_utts

            logger.info(
                'Epoch {} Val info val_loss {}'.format(self.epoch, cv_loss))
            if self.visualizer:
                self.visualizer.add_scalar(
                    tag='eval/cv_loss', value=cv_loss, step=self.epoch)
                self.visualizer.add_scalar(
                    tag='eval/lr', value=self.lr_scheduler(), step=self.epoch)

            self.save(tag=self.epoch, infos={'val_loss': cv_loss})
            self.new_epoch()

    def setup_dataloader(self):
        self.use_streamdata = config.get("use_stream_data", False)
        if self.train:
            config = self.config.clone()
            self.train_loader = DataLoaderFactory.get_dataloader(
                'train', config, self.args)
            config = self.config.clone()
            config['preprocess_config'] = None
            self.valid_loader = DataLoaderFactory.get_dataloader(
                'valid', config, self.args)
            logger.info("Setup train/valid Dataloader!")
        else:
            config = self.config.clone()
            config['preprocess_config'] = None
            self.test_loader = DataLoaderFactory.get_dataloader('test', config,
                                                                self.args)
            config = self.config.clone()
            config['preprocess_config'] = None
            self.align_loader = DataLoaderFactory.get_dataloader(
                'align', config, self.args)
            logger.info("Setup test/align Dataloader!")

    def setup_model(self):
        config = self.config

        # model
        model_conf = config
        with UpdateConfig(model_conf):
            model_conf.input_dim = self.train_loader.feat_dim
            model_conf.output_dim = self.train_loader.vocab_size
        model = U2Model.from_config(model_conf)
        if self.parallel:
            model = paddle.DataParallel(model)
        layer_tools.print_params(model, logger.info)

        # lr
        scheduler_conf = config.scheduler_conf
        scheduler_args = {
            "learning_rate": scheduler_conf.lr,
            "warmup_steps": scheduler_conf.warmup_steps,
            "gamma": scheduler_conf.lr_decay,
            "d_model": model_conf.encoder_conf.output_size,
            "verbose": False,
        }
        lr_scheduler = LRSchedulerFactory.from_args(config.scheduler,
                                                    scheduler_args)

        # opt
        def optimizer_args(
                config,
                parameters,
                lr_scheduler=None, ):
            optim_conf = config.optim_conf
            return {
                "grad_clip": optim_conf.global_grad_clip,
                "weight_decay": optim_conf.weight_decay,
                "learning_rate": lr_scheduler,
                "parameters": parameters,
            }

        optimzer_args = optimizer_args(config, model.parameters(), lr_scheduler)
        optimizer = OptimizerFactory.from_args(config.optim, optimzer_args)

        self.model = model
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        logger.info("Setup model/optimizer/lr_scheduler!")


class U2Tester(U2Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.text_feature = TextFeaturizer(
            unit_type=self.config.unit_type,
            vocab=self.config.vocab_filepath,
            spm_model_prefix=self.config.spm_model_prefix)
        self.vocab_list = self.text_feature.vocab_list

    def id2token(self, texts, texts_len, text_feature):
        """ ord() id to chr() chr """
        trans = []
        for text, n in zip(texts, texts_len):
            n = n.numpy().item()
            ids = text[:n]
            trans.append(text_feature.defeaturize(ids.numpy().tolist()))
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

        start_time = time.time()
        target_transcripts = self.id2token(texts, texts_len, self.text_feature)
        result_transcripts, result_tokenids = self.model.decode(
            audio,
            audio_len,
            text_feature=self.text_feature,
            decoding_method=decode_cfg.decoding_method,
            beam_size=decode_cfg.beam_size,
            ctc_weight=decode_cfg.ctc_weight,
            decoding_chunk_size=decode_cfg.decoding_chunk_size,
            num_decoding_left_chunks=decode_cfg.num_decoding_left_chunks,
            simulate_streaming=decode_cfg.simulate_streaming)
        decode_time = time.time() - start_time

        for i, (utt, target, result, rec_tids) in enumerate(
                zip(utts, target_transcripts, result_transcripts,
                    result_tokenids)):
            errors, len_ref = errors_func(target, result)
            errors_sum += errors
            len_refs += len_ref
            num_ins += 1
            if fout:
                fout.write({
                    "utt": utt,
                    "refs": [target],
                    "hyps": [result],
                    "hyps_tokenid": [rec_tids],
                })
            logger.info(f"Utt: {utt}")
            logger.info(f"Ref: {target}")
            logger.info(f"Hyp: {result}")
            logger.info(
                "One example error rate [%s] = %f" %
                (decode_cfg.error_rate_type, error_rate_func(target, result)))

        return dict(
            errors_sum=errors_sum,
            len_refs=len_refs,
            num_ins=num_ins,  # num examples
            error_rate=errors_sum / len_refs,
            error_rate_type=decode_cfg.error_rate_type,
            num_frames=audio_len.sum().numpy().item(),
            decode_time=decode_time)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        assert self.args.result_file
        self.model.eval()
        if not self.use_streamdata:
            logger.info(f"Test Total Examples: {len(self.test_loader.dataset)}")

        stride_ms = self.config.stride_ms
        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        num_frames = 0.0
        num_time = 0.0
        with jsonlines.open(self.args.result_file, 'w') as fout:
            for i, batch in enumerate(self.test_loader):
                metrics = self.compute_metrics(*batch, fout=fout)
                num_frames += metrics['num_frames']
                num_time += metrics["decode_time"]
                errors_sum += metrics['errors_sum']
                len_refs += metrics['len_refs']
                num_ins += metrics['num_ins']
                error_rate_type = metrics['error_rate_type']
                rtf = num_time / (num_frames * stride_ms)
                logger.info(
                    "RTF: %f, Error rate [%s] (%d/?) = %f" %
                    (rtf, error_rate_type, num_ins, errors_sum / len_refs))

        rtf = num_time / (num_frames * stride_ms)
        msg = "Test: "
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += "RTF: {}, ".format(rtf)
        msg += "Final error rate [%s] (%d/%d) = %f" % (
            error_rate_type, num_ins, num_ins, errors_sum / len_refs)
        logger.info(msg)

        # test meta results
        err_meta_path = os.path.splitext(self.args.result_file)[0] + '.err'
        err_type_str = "{}".format(error_rate_type)
        with open(err_meta_path, 'w') as f:
            data = json.dumps({
                "epoch":
                self.epoch,
                "step":
                self.iteration,
                "rtf":
                rtf,
                error_rate_type:
                errors_sum / len_refs,
                "dataset_hour": (num_frames * stride_ms) / 1000.0 / 3600.0,
                "process_hour":
                num_time / 1000.0 / 3600.0,
                "num_examples":
                num_ins,
                "err_sum":
                errors_sum,
                "ref_len":
                len_refs,
                "decode_method":
                self.config.decode.decoding_method,
            })
            f.write(data + '\n')

    @paddle.no_grad()
    def align(self):
        ctc_utils.ctc_align(self.config, self.model, self.align_loader,
                            self.config.decode.decode_batch_size,
                            self.config.stride_ms, self.vocab_list,
                            self.args.result_file)

    def load_inferspec(self):
        """infer model and input spec.

        Returns:
            nn.Layer: inference model
            List[paddle.static.InputSpec]: input spec.
        """
        from paddlespeech.s2t.models.u2 import U2InferModel
        infer_model = U2InferModel.from_pretrained(self.test_loader,
                                                   self.config.clone(),
                                                   self.args.checkpoint_path)
        feat_dim = self.test_loader.feat_dim
        input_spec = [
            paddle.static.InputSpec(shape=[1, None, feat_dim],
                                    dtype='float32'),  # audio, [B,T,D]
            paddle.static.InputSpec(shape=[1],
                                    dtype='int64'),  # audio_length, [B]
        ]
        return infer_model, input_spec

    @paddle.no_grad()
    def export(self):
        infer_model, input_spec = self.load_inferspec()
        assert isinstance(input_spec, list), type(input_spec)
        infer_model.eval()
        static_model = paddle.jit.to_static(infer_model, input_spec=input_spec)
        logger.info(f"Export code: {static_model.forward.code}")
        paddle.jit.save(static_model, self.args.export_path)

    def setup_dict(self):
        # load dictionary for debug log
        self.args.char_list = load_dict(self.args.dict_path,
                                        "maskctc" in self.args.model_name)

    def setup(self):
        super().setup()
        self.setup_dict()
