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
from collections import OrderedDict
from contextlib import nullcontext

import jsonlines
import numpy as np
import paddle
from paddle import distributed as dist

from paddlespeech.s2t.frontend.featurizer import TextFeaturizer
from paddlespeech.s2t.io.dataloader import DataLoaderFactory
from paddlespeech.s2t.models.u2_st import U2STModel
from paddlespeech.s2t.training.optimizer import OptimizerFactory
from paddlespeech.s2t.training.reporter import ObsScope
from paddlespeech.s2t.training.reporter import report
from paddlespeech.s2t.training.scheduler import LRSchedulerFactory
from paddlespeech.s2t.training.timer import Timer
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils import bleu_score
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig

logger = Log(__name__).getlog()


class U2STTrainer(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)

    def train_batch(self, batch_index, batch_data, msg):
        train_conf = self.config
        start = time.time()
        # forward
        utt, audio, audio_len, text, text_len = batch_data
        if isinstance(text, list) and isinstance(text_len, list):
            # joint training with ASR. Two decoding texts [translation, transcription]
            text, text_transcript = text
            text_len, text_transcript_len = text_len
            loss, st_loss, attention_loss, ctc_loss = self.model(
                audio, audio_len, text, text_len, text_transcript,
                text_transcript_len)
        else:
            loss, st_loss, attention_loss, ctc_loss = self.model(
                audio, audio_len, text, text_len)

        # loss div by `batch_size * accum_grad`
        loss /= train_conf.accum_grad
        losses_np = {'loss': float(loss) * train_conf.accum_grad}
        if st_loss:
            losses_np['st_loss'] = float(st_loss)
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

        for k, v in losses_np.items():
            report(k, v)
        report("batch_size", self.config.batch_size)
        report("accum", train_conf.accum_grad)
        report("step_cost", iteration_time)

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
            logger.info(f"Valid Total Examples: {len(self.valid_loader.dataset)}")
        valid_losses = defaultdict(list)
        num_seen_utts = 1
        total_loss = 0.0
        for i, batch in enumerate(self.valid_loader):
            utt, audio, audio_len, text, text_len = batch
            if isinstance(text, list) and isinstance(text_len, list):
                text, text_transcript = text
                text_len, text_transcript_len = text_len
                loss, st_loss, attention_loss, ctc_loss = self.model(
                    audio, audio_len, text, text_len, text_transcript,
                    text_transcript_len)
            else:
                loss, st_loss, attention_loss, ctc_loss = self.model(
                    audio, audio_len, text, text_len)
            if paddle.isfinite(loss):
                num_utts = batch[1].shape[0]
                num_seen_utts += num_utts
                total_loss += float(st_loss) * num_utts
                valid_losses['val_loss'].append(float(st_loss))
                if attention_loss:
                    valid_losses['val_att_loss'].append(float(attention_loss))
                if ctc_loss:
                    valid_losses['val_ctc_loss'].append(float(ctc_loss))

            if (i + 1) % self.config.log_interval == 0:
                valid_dump = {k: np.mean(v) for k, v in valid_losses.items()}
                valid_dump['val_history_st_loss'] = total_loss / num_seen_utts

                # logging
                msg = f"Valid: Rank: {dist.get_rank()}, "
                msg += "epoch: {}, ".format(self.epoch)
                msg += "step: {}, ".format(self.iteration)
                if not self.use_streamdata:
                    msg += "batch: {}/{}, ".format(i + 1, len(self.valid_loader))
                msg += ', '.join('{}: {:>.6f}'.format(k, v)
                                 for k, v in valid_dump.items())
                logger.info(msg)

        logger.info('Rank {} Val info st_val_loss {}'.format(
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
            logger.info(f"Train Total Examples: {len(self.train_loader.dataset)}")
        while self.epoch < self.config.n_epoch:
            with Timer("Epoch-Train Time Cost: {}"):
                self.model.train()
                try:
                    data_start_time = time.time()
                    for batch_index, batch in enumerate(self.train_loader):
                        dataload_time = time.time() - data_start_time
                        msg = "Train:"
                        observation = OrderedDict()
                        with ObsScope(observation):
                            report("Rank", dist.get_rank())
                            report("epoch", self.epoch)
                            report('step', self.iteration)
                            report("lr", self.lr_scheduler())
                            self.train_batch(batch_index, batch, msg)
                            self.after_train_batch()
                            report('iter', batch_index + 1)
                            if not self.use_streamdata:
                                report('total', len(self.train_loader))
                            report('reader_cost', dataload_time)
                        observation['batch_cost'] = observation[
                            'reader_cost'] + observation['step_cost']
                        observation['samples'] = observation['batch_size']
                        observation['ips,sent./sec'] = observation[
                            'batch_size'] / observation['batch_cost']
                        for k, v in observation.items():
                            msg += f" {k.split(',')[0]}: "
                            msg += f"{v:>.8f}" if isinstance(v,
                                                             float) else f"{v}"
                            msg += f" {k.split(',')[1]}" if len(
                                k.split(',')) == 2 else ""
                            msg += ","
                        msg = msg[:-1]  # remove the last ","
                        if (batch_index + 1) % self.config.log_interval == 0:
                            logger.info(msg)
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
        config = self.config.clone()

        load_transcript = True if config.model_conf.asr_weight > 0 else False

        config = self.config.clone()
        config['load_transcript'] = load_transcript
        self.use_streamdata = config.get("use_stream_data", False)
        if self.train:
            self.train_loader = DataLoaderFactory.get_dataloader('train', config, self.args)
            self.valid_loader = DataLoaderFactory.get_dataloader('valid', config, self.args)
            logger.info("Setup train/valid Dataloader!")
        else:
            self.test_loader = DataLoaderFactory.get_dataloader('test', config, self.args)
            logger.info("Setup test Dataloader!")


    def setup_model(self):
        config = self.config
        model_conf = config
        with UpdateConfig(model_conf):
            if self.train:
                model_conf.input_dim = self.train_loader.feat_dim
                model_conf.output_dim = self.train_loader.vocab_size
            else:
                model_conf.input_dim = self.test_loader.feat_dim
                model_conf.output_dim = self.test_loader.vocab_size

        model = U2STModel.from_config(model_conf)

        if self.parallel:
            model = paddle.DataParallel(model)

        logger.info(f"{model}")
        layer_tools.print_params(model, logger.info)

        train_config = config
        optim_type = train_config.optim
        optim_conf = train_config.optim_conf
        scheduler_type = train_config.scheduler
        scheduler_conf = train_config.scheduler_conf

        scheduler_args = {
            "learning_rate": optim_conf.lr,
            "verbose": False,
            "warmup_steps": scheduler_conf.warmup_steps,
            "gamma": scheduler_conf.lr_decay,
            "d_model": model_conf.encoder_conf.output_size,
        }
        lr_scheduler = LRSchedulerFactory.from_args(scheduler_type,
                                                    scheduler_args)

        def optimizer_args(
                config,
                parameters,
                lr_scheduler=None, ):
            train_config = config
            optim_type = train_config.optim
            optim_conf = train_config.optim_conf
            scheduler_type = train_config.scheduler
            scheduler_conf = train_config.scheduler_conf
            return {
                "grad_clip": train_config.global_grad_clip,
                "weight_decay": optim_conf.weight_decay,
                "learning_rate": lr_scheduler
                if lr_scheduler else optim_conf.lr,
                "parameters": parameters,
                "epsilon": 1e-9 if optim_type == 'noam' else None,
                "beta1": 0.9 if optim_type == 'noam' else None,
                "beat2": 0.98 if optim_type == 'noam' else None,
            }

        optimzer_args = optimizer_args(config, model.parameters(), lr_scheduler)
        optimizer = OptimizerFactory.from_args(optim_type, optimzer_args)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        logger.info("Setup model/optimizer/lr_scheduler!")


class U2STTester(U2STTrainer):
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

    def translate(self, audio, audio_len):
        """"E2E translation from extracted audio feature"""
        decode_cfg = self.config.decode
        self.model.eval()

        hyps = self.model.decode(
            audio,
            audio_len,
            text_feature=self.text_feature,
            decoding_method=decode_cfg.decoding_method,
            beam_size=decode_cfg.beam_size,
            word_reward=decode_cfg.word_reward,
            maxlenratio=decode_cfg.maxlenratio,
            decoding_chunk_size=decode_cfg.decoding_chunk_size,
            num_decoding_left_chunks=decode_cfg.num_decoding_left_chunks,
            simulate_streaming=decode_cfg.simulate_streaming)
        return hyps

    def compute_translation_metrics(self,
                                    utts,
                                    audio,
                                    audio_len,
                                    texts,
                                    texts_len,
                                    bleu_func,
                                    fout=None):
        decode_cfg = self.config.decode
        len_refs, num_ins = 0, 0

        start_time = time.time()

        refs = self.id2token(texts, texts_len, self.text_feature)

        hyps = self.model.decode(
            audio,
            audio_len,
            text_feature=self.text_feature,
            decoding_method=decode_cfg.decoding_method,
            beam_size=decode_cfg.beam_size,
            word_reward=decode_cfg.word_reward,
            maxlenratio=decode_cfg.maxlenratio,
            decoding_chunk_size=decode_cfg.decoding_chunk_size,
            num_decoding_left_chunks=decode_cfg.num_decoding_left_chunks,
            simulate_streaming=decode_cfg.simulate_streaming)

        decode_time = time.time() - start_time

        for utt, target, result in zip(utts, refs, hyps):
            len_refs += len(target.split())
            num_ins += 1
            if fout:
                fout.write({"utt": utt, "ref": target, "hyp": result})
            logger.info(f"Utt: {utt}")
            logger.info(f"Ref: {target}")
            logger.info(f"Hyp: {result}")
            logger.info("One example BLEU = %s" %
                        (bleu_func([result], [[target]]).prec_str))

        return dict(
            hyps=hyps,
            refs=refs,
            bleu=bleu_func(hyps, [refs]).score,
            len_refs=len_refs,
            num_ins=num_ins,  # num examples
            num_frames=audio_len.sum().numpy().item(),
            decode_time=decode_time)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        assert self.args.result_file
        self.model.eval()
        if not self.use_streamdata:
            logger.info(f"Test Total Examples: {len(self.test_loader.dataset)}")

        decode_cfg = self.config.decode
        bleu_func = bleu_score.char_bleu if decode_cfg.error_rate_type == 'char-bleu' else bleu_score.bleu

        stride_ms = self.config.stride_ms
        hyps, refs = [], []
        len_refs, num_ins = 0, 0
        num_frames = 0.0
        num_time = 0.0
        with jsonlines.open(self.args.result_file, 'w') as fout:
            for i, batch in enumerate(self.test_loader):
                metrics = self.compute_translation_metrics(
                    *batch, bleu_func=bleu_func, fout=fout)
                hyps += metrics['hyps']
                refs += metrics['refs']
                bleu = metrics['bleu']
                num_frames += metrics['num_frames']
                num_time += metrics["decode_time"]
                len_refs += metrics['len_refs']
                num_ins += metrics['num_ins']
                rtf = num_time / (num_frames * stride_ms)
                logger.info("RTF: %f, instance (%d), batch BELU   = %f" %
                            (rtf, num_ins, bleu))

        rtf = num_time / (num_frames * stride_ms)
        msg = "Test: "
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += "RTF: {}, ".format(rtf)
        msg += "Test set [%s]: %s" % (len(hyps), str(bleu_func(hyps, [refs])))
        logger.info(msg)
        bleu_meta_path = os.path.splitext(self.args.result_file)[0] + '.bleu'
        err_type_str = "BLEU"
        with open(bleu_meta_path, 'w') as f:
            data = json.dumps({
                "epoch":
                self.epoch,
                "step":
                self.iteration,
                "rtf":
                rtf,
                err_type_str:
                bleu_func(hyps, [refs]).score,
                "dataset_hour": (num_frames * stride_ms) / 1000.0 / 3600.0,
                "process_hour":
                num_time / 1000.0 / 3600.0,
                "num_examples":
                num_ins,
                "decode_method":
                self.config.decode.decoding_method,
            })
            f.write(data + '\n')

    def load_inferspec(self):
        """infer model and input spec.

        Returns:
            nn.Layer: inference model
            List[paddle.static.InputSpec]: input spec.
        """
        from paddlespeech.s2t.models.u2_st import U2STInferModel
        infer_model = U2STInferModel.from_pretrained(self.test_loader,
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
