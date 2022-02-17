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
from contextlib import nullcontext

import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader
from src_deepspeech2x.models.ds2 import DeepSpeech2InferModel
from src_deepspeech2x.models.ds2 import DeepSpeech2Model

from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.io.dataset import ManifestDataset
from paddlespeech.s2t.io.sampler import SortagradBatchSampler
from paddlespeech.s2t.io.sampler import SortagradDistributedBatchSampler
from paddlespeech.s2t.models.ds2_online import DeepSpeech2InferModelOnline
from paddlespeech.s2t.models.ds2_online import DeepSpeech2ModelOnline
from paddlespeech.s2t.training.gradclip import ClipGradByGlobalNormWithLog
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils import error_rate
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


class DeepSpeech2Trainer(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)

    def train_batch(self, batch_index, batch_data, msg):
        train_conf = self.config
        start = time.time()

        # forward
        utt, audio, audio_len, text, text_len = batch_data
        loss = self.model(audio, audio_len, text, text_len)
        losses_np = {
            'train_loss': float(loss),
        }

        # loss backward
        if (batch_index + 1) % train_conf.accum_grad != 0:
            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            context = self.model.no_sync
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
            self.iteration += 1

        iteration_time = time.time() - start

        msg += "train time: {:>.3f}s, ".format(iteration_time)
        msg += "batch size: {}, ".format(self.config.batch_size)
        msg += "accum: {}, ".format(train_conf.accum_grad)
        msg += ', '.join('{}: {:>.6f}'.format(k, v)
                         for k, v in losses_np.items())
        logger.info(msg)

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
        config.defrost()
        config.feat_size = self.train_loader.collate_fn.feature_size
        #config.dict_size = self.train_loader.collate_fn.vocab_size
        config.dict_size = len(self.train_loader.collate_fn.vocab_list)
        config.freeze()

        if self.args.model_type == 'offline':
            model = DeepSpeech2Model.from_config(config)
        elif self.args.model_type == 'online':
            model = DeepSpeech2ModelOnline.from_config(config)
        else:
            raise Exception("wrong model type")
        if self.parallel:
            model = paddle.DataParallel(model)

        logger.info(f"{model}")
        layer_tools.print_params(model, logger.info)

        grad_clip = ClipGradByGlobalNormWithLog(config.global_grad_clip)
        lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=config.lr, gamma=config.lr_decay, verbose=True)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=paddle.regularizer.L2Decay(config.weight_decay),
            grad_clip=grad_clip)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        logger.info("Setup model/optimizer/lr_scheduler!")

    def setup_dataloader(self):
        config = self.config.clone()
        config.defrost()
        config.keep_transcription_text = False

        config.manifest = config.train_manifest
        train_dataset = ManifestDataset.from_config(config)

        config.manifest = config.dev_manifest
        dev_dataset = ManifestDataset.from_config(config)

        config.manifest = config.test_manifest
        test_dataset = ManifestDataset.from_config(config)

        if self.parallel:
            batch_sampler = SortagradDistributedBatchSampler(
                train_dataset,
                batch_size=config.batch_size,
                num_replicas=None,
                rank=None,
                shuffle=True,
                drop_last=True,
                sortagrad=config.sortagrad,
                shuffle_method=config.shuffle_method)
        else:
            batch_sampler = SortagradBatchSampler(
                train_dataset,
                shuffle=True,
                batch_size=config.batch_size,
                drop_last=True,
                sortagrad=config.sortagrad,
                shuffle_method=config.shuffle_method)

        collate_fn_train = SpeechCollator.from_config(config)

        config.augmentation_config = ""
        collate_fn_dev = SpeechCollator.from_config(config)

        config.keep_transcription_text = True
        config.augmentation_config = ""
        collate_fn_test = SpeechCollator.from_config(config)

        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn_train,
            num_workers=config.num_workers)
        self.valid_loader = DataLoader(
            dev_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn_dev)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.decode.decode_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn_test)
        if "<eos>" in self.test_loader.collate_fn.vocab_list:
            self.test_loader.collate_fn.vocab_list.remove("<eos>")
        if "<eos>" in self.valid_loader.collate_fn.vocab_list:
            self.valid_loader.collate_fn.vocab_list.remove("<eos>")
        if "<eos>" in self.train_loader.collate_fn.vocab_list:
            self.train_loader.collate_fn.vocab_list.remove("<eos>")
        logger.info("Setup train/valid/test  Dataloader!")


class DeepSpeech2Tester(DeepSpeech2Trainer):
    def __init__(self, config, args):

        self._text_featurizer = TextFeaturizer(
            unit_type=config.unit_type, vocab=None)
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
        cfg = self.config.decode
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        errors_func = error_rate.char_errors if cfg.error_rate_type == 'cer' else error_rate.word_errors
        error_rate_func = error_rate.cer if cfg.error_rate_type == 'cer' else error_rate.wer

        target_transcripts = self.ordid2token(texts, texts_len)

        result_transcripts = self.compute_result_transcripts(audio, audio_len)

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

    def compute_result_transcripts(self, audio, audio_len):
        result_transcripts = self.model.decode(audio, audio_len)

        result_transcripts = [
            self._text_featurizer.detokenize(item)
            for item in result_transcripts
        ]
        return result_transcripts

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        logger.info(f"Test Total Examples: {len(self.test_loader.dataset)}")
        self.model.eval()
        cfg = self.config
        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0

        # Initialized the decoder in model
        decode_cfg = self.config.decode
        vocab_list = self.test_loader.collate_fn.vocab_list
        decode_batch_size = self.test_loader.batch_size
        self.model.decoder.init_decoder(
            decode_batch_size, vocab_list, decode_cfg.decoding_method,
            decode_cfg.lang_model_path, decode_cfg.alpha, decode_cfg.beta,
            decode_cfg.beam_size, decode_cfg.cutoff_prob,
            decode_cfg.cutoff_top_n, decode_cfg.num_proc_bsearch)

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
        self.model.decoder.del_decoder()

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
