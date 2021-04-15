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

import sys
import time
from deepspeech.utils.log import Log
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Optional
from yacs.config import CfgNode

import paddle
from paddle import distributed as dist
from paddle.io import DataLoader

from deepspeech.training.trainer import Trainer
from deepspeech.training.gradclip import ClipGradByGlobalNormWithLog
from deepspeech.training.scheduler import WarmupLR

from deepspeech.utils import mp_tools
from deepspeech.utils import layer_tools
from deepspeech.utils import error_rate

from deepspeech.io.collator import SpeechCollator
from deepspeech.io.sampler import SortagradDistributedBatchSampler
from deepspeech.io.sampler import SortagradBatchSampler
from deepspeech.io.dataset import ManifestDataset

from deepspeech.models.u2 import U2Model

logger = Log(__name__).getlog()


class U2Trainer(Trainer):
    @classmethod
    def params(cls, config: Optional[CfgNode]=None) -> CfgNode:
        # training config
        default = CfgNode(
            dict(
                n_epoch=50,  # train epochs
                log_interval=100,  # steps
                accum_grad=1,  # accum grad by # steps
                global_grad_clip=5.0,  # the global norm clip
            ))
        default.optim = 'adam'
        default.optim_conf = CfgNode(
            dict(
                lr=5e-4,  # learning rate
                weight_decay=1e-6,  # the coeff of weight decay
            ))
        default.scheduler = 'warmuplr'
        default.scheduler_conf = CfgNode(
            dict(
                warmup_steps=25000,
                lr_decay=1.0,  # learning rate decay
            ))

        if config is not None:
            config.merge_from_other_cfg(default)
        return default

    def __init__(self, config, args):
        super().__init__(config, args)

    def train_batch(self, batch_index, batch_data, msg):
        train_conf = self.config.training
        start = time.time()

        loss, attention_loss, ctc_loss = self.model(*batch_data)
        # loss div by `batch_size * accum_grad`
        loss /= train_conf.accum_grad
        loss.backward()
        layer_tools.print_grads(self.model, print_func=None)

        losses_np = {
            'train_loss': float(loss) * train_conf.accum_grad,
            'train_att_loss': float(attention_loss),
            'train_ctc_loss': float(ctc_loss),
        }

        if (batch_index + 1) % train_conf.accum_grad == 0:
            if dist.get_rank() == 0 and self.visualizer:
                for k, v in losses_np.items():
                    self.visualizer.add_scalar("train/{}".format(k), v,
                                               self.iteration)
            self.optimizer.step()
            self.optimizer.clear_grad()
            self.lr_scheduler.step()
            self.iteration += 1

        iteration_time = time.time() - start

        if (batch_index + 1) % train_conf.log_interval == 0:
            msg += "time: {:>.3f}s, ".format(iteration_time)
            msg += "batch size: {}, ".format(self.config.data.batch_size)
            msg += "accum: {}, ".format(train_conf.accum_grad)
            msg += ', '.join('{}: {:>.6f}'.format(k, v)
                             for k, v in losses_np.items())
            self.logger.info(msg)

    def train(self):
        """The training process control by step."""
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        # script_model = paddle.jit.to_static(self.model)
        # script_model_path = str(self.checkpoint_dir / 'init')
        # paddle.jit.save(script_model, script_model_path)

        from_scratch = self.resume_or_scratch()
        if from_scratch:
            # save init model, i.e. 0 epoch
            self.save(tag='init')

        self.lr_scheduler.step(self.iteration)
        if self.parallel:
            self.train_loader.batch_sampler.set_epoch(self.epoch)

        self.logger.info(
            f"Train Total Examples: {len(self.train_loader.dataset)}")
        while self.epoch < self.config.training.n_epoch:
            self.model.train()
            try:
                data_start_time = time.time()
                for batch_index, batch in enumerate(self.train_loader):
                    dataload_time = time.time() - data_start_time
                    msg = "Train: Rank: {}, ".format(dist.get_rank())
                    msg += "epoch: {}, ".format(self.epoch)
                    msg += "step: {}, ".format(self.iteration)
                    msg += "lr: {:>.8f}, ".format(self.lr_scheduler())
                    msg += "dataloader time: {:>.3f}s, ".format(dataload_time)
                    self.train_batch(batch_index, batch, msg)
                    data_start_time = time.time()
            except Exception as e:
                self.logger.error(e)
                raise e

            valid_losses = self.valid()
            self.save(infos=valid_losses)
            self.new_epoch()

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        self.model.eval()
        self.logger.info(
            f"Valid Total Examples: {len(self.valid_loader.dataset)}")
        valid_losses = defaultdict(list)
        for i, batch in enumerate(self.valid_loader):
            total_loss, attention_loss, ctc_loss = self.model(*batch)

            valid_losses['val_loss'].append(float(total_loss))
            valid_losses['val_att_loss'].append(float(attention_loss))
            valid_losses['val_ctc_loss'].append(float(ctc_loss))

        # write visual log
        valid_losses = {k: np.mean(v) for k, v in valid_losses.items()}

        # logging
        msg = f"Valid: Rank: {dist.get_rank()}, "
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += ', '.join('{}: {:>.6f}'.format(k, v)
                         for k, v in valid_losses.items())
        self.logger.info(msg)

        if self.visualizer:
            for k, v in valid_losses.items():
                self.visualizer.add_scalar("valid/{}".format(k), v,
                                           self.iteration)
        return valid_losses

    def setup_dataloader(self):
        config = self.config.clone()
        config.defrost()
        config.data.keep_transcription_text = False

        # train/valid dataset, return token ids
        config.data.manifest = config.data.train_manifest
        train_dataset = ManifestDataset.from_config(config)

        config.data.manifest = config.data.dev_manifest
        config.data.augmentation_config = ""
        dev_dataset = ManifestDataset.from_config(config)

        collate_fn = SpeechCollator(keep_transcription_text=False)
        if self.parallel:
            batch_sampler = SortagradDistributedBatchSampler(
                train_dataset,
                batch_size=config.data.batch_size,
                num_replicas=None,
                rank=None,
                shuffle=True,
                drop_last=True,
                sortagrad=config.data.sortagrad,
                shuffle_method=config.data.shuffle_method)
        else:
            batch_sampler = SortagradBatchSampler(
                train_dataset,
                shuffle=True,
                batch_size=config.data.batch_size,
                drop_last=True,
                sortagrad=config.data.sortagrad,
                shuffle_method=config.data.shuffle_method)
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=config.data.num_workers, )
        self.valid_loader = DataLoader(
            dev_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        # test dataset, return raw text
        config.data.keep_transcription_text = True
        config.data.augmentation_config = ""
        config.data.manifest = config.data.test_manifest
        test_dataset = ManifestDataset.from_config(config)
        # return text ord id
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.decoding.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=SpeechCollator(keep_transcription_text=True))
        self.logger.info("Setup train/valid/test Dataloader!")

    def setup_model(self):
        config = self.config
        model_conf = config.model
        model_conf.defrost()
        model_conf.input_dim = self.train_loader.dataset.feature_size
        model_conf.output_dim = self.train_loader.dataset.vocab_size
        model_conf.freeze()
        model = U2Model.from_config(model_conf)

        if self.parallel:
            model = paddle.DataParallel(model)

        layer_tools.print_params(model, self.logger.info)

        train_config = config.training
        optim_type = train_config.optim
        optim_conf = train_config.optim_conf
        scheduler_type = train_config.scheduler
        scheduler_conf = train_config.scheduler_conf

        grad_clip = ClipGradByGlobalNormWithLog(train_config.global_grad_clip)
        weight_decay = paddle.regularizer.L2Decay(optim_conf.weight_decay)

        if scheduler_type == 'expdecaylr':
            lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
                learning_rate=optim_conf.lr,
                gamma=scheduler_conf.lr_decay,
                verbose=False)
        elif scheduler_type == 'warmuplr':
            lr_scheduler = WarmupLR(
                learning_rate=optim_conf.lr,
                warmup_steps=scheduler_conf.warmup_steps,
                verbose=False)
        else:
            raise ValueError(f"Not support scheduler: {scheduler_type}")

        if optim_type == 'adam':
            optimizer = paddle.optimizer.Adam(
                learning_rate=lr_scheduler,
                parameters=model.parameters(),
                weight_decay=weight_decay,
                grad_clip=grad_clip)
        else:
            raise ValueError(f"Not support optim: {optim_type}")

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger.info("Setup model/optimizer/lr_scheduler!")


class U2Tester(U2Trainer):
    @classmethod
    def params(cls, config: Optional[CfgNode]=None) -> CfgNode:
        # decoding config
        default = CfgNode(
            dict(
                alpha=2.5,  # Coef of LM for beam search.
                beta=0.3,  # Coef of WC for beam search.
                cutoff_prob=1.0,  # Cutoff probability for pruning.
                cutoff_top_n=40,  # Cutoff number for pruning.
                lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm',  # Filepath for language model.
                decoding_method='attention',  # Decoding method. Options: 'attention', 'ctc_greedy_search',
                # 'ctc_prefix_beam_search', 'attention_rescoring'
                error_rate_type='wer',  # Error rate type for evaluation. Options `wer`, 'cer'
                num_proc_bsearch=8,  # # of CPUs for beam search.
                beam_size=10,  # Beam search width.
                batch_size=16,  # decoding batch size
                ctc_weight=0.0,  # ctc weight for attention rescoring decode mode.
                decoding_chunk_size=-1,  # decoding chunk size. Defaults to -1.
                # <0: for decoding, use full chunk.
                # >0: for decoding, use fixed chunk size as set.
                # 0: used for training, it's prohibited here. 
                num_decoding_left_chunks=-1,  # number of left chunks for decoding. Defaults to -1.
                simulate_streaming=False,  # simulate streaming inference. Defaults to False.
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

    def compute_metrics(self, audio, audio_len, texts, texts_len, fout=None):
        cfg = self.config.decoding
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        errors_func = error_rate.char_errors if cfg.error_rate_type == 'cer' else error_rate.word_errors
        error_rate_func = error_rate.cer if cfg.error_rate_type == 'cer' else error_rate.wer

        text_feature = self.test_loader.dataset.text_feature

        target_transcripts = self.ordid2token(texts, texts_len)
        result_transcripts = self.model.decode(
            audio,
            audio_len,
            text_feature=text_feature,
            decoding_method=cfg.decoding_method,
            lang_model_path=cfg.lang_model_path,
            beam_alpha=cfg.alpha,
            beam_beta=cfg.beta,
            beam_size=cfg.beam_size,
            cutoff_prob=cfg.cutoff_prob,
            cutoff_top_n=cfg.cutoff_top_n,
            num_processes=cfg.num_proc_bsearch,
            ctc_weight=cfg.ctc_weight,
            decoding_chunk_size=cfg.decoding_chunk_size,
            num_decoding_left_chunks=cfg.num_decoding_left_chunks,
            simulate_streaming=cfg.simulate_streaming)

        for target, result in zip(target_transcripts, result_transcripts):
            errors, len_ref = errors_func(target, result)
            errors_sum += errors
            len_refs += len_ref
            num_ins += 1
            if fout:
                fout.write(result + "\n")
            self.logger.info(
                "\nTarget Transcription: %s\nOutput Transcription: %s" %
                (target, result))
            self.logger.info("Current error rate [%s] = %f" % (
                cfg.error_rate_type, error_rate_func(target, result)))

        return dict(
            errors_sum=errors_sum,
            len_refs=len_refs,
            num_ins=num_ins,
            error_rate=errors_sum / len_refs,
            error_rate_type=cfg.error_rate_type)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        assert self.args.result_file
        self.model.eval()
        self.logger.info(
            f"Test Total Examples: {len(self.test_loader.dataset)}")

        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0

        with open(self.args.result_file, 'w') as fout:
            for i, batch in enumerate(self.test_loader):
                metrics = self.compute_metrics(*batch, fout=fout)
                errors_sum += metrics['errors_sum']
                len_refs += metrics['len_refs']
                num_ins += metrics['num_ins']
                error_rate_type = metrics['error_rate_type']
                self.logger.info(
                    "Error rate [%s] (%d/?) = %f" %
                    (error_rate_type, num_ins, errors_sum / len_refs))

        # logging
        msg = "Test: "
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += ", Final error rate [%s] (%d/%d) = %f" % (
            error_rate_type, num_ins, num_ins, errors_sum / len_refs)
        self.logger.info(msg)

    def run_test(self):
        self.resume_or_scratch()
        try:
            self.test()
        except KeyboardInterrupt:
            sys.exit(-1)

    def load_inferspec(self):
        """infer model and input spec.

        Returns:
            nn.Layer: inference model
            List[paddle.static.InputSpec]: input spec.
        """
        from deepspeech.models.u2 import U2InferModel
        infer_model = U2InferModel.from_pretrained(self.test_loader.dataset,
                                                   self.config.model.clone(),
                                                   self.args.checkpoint_path)
        feat_dim = self.test_loader.dataset.feature_size
        input_spec = [
            paddle.static.InputSpec(
                shape=[None, feat_dim, None],
                dtype='float32'),  # audio, [B,D,T]
            paddle.static.InputSpec(shape=[None],
                                    dtype='int64'),  # audio_length, [B]
        ]
        return infer_model, input_spec

    def export(self):
        infer_model, input_spec = self.load_inferspec()
        assert isinstance(input_spec, list), type(input_spec)
        infer_model.eval()
        static_model = paddle.jit.to_static(infer_model, input_spec=input_spec)
        logger.info(f"Export code: {static_model.forward.code}")
        paddle.jit.save(static_model, self.args.export_path)

    def run_export(self):
        try:
            self.export()
        except KeyboardInterrupt:
            sys.exit(-1)

    def setup(self):
        """Setup the experiment.
        """
        paddle.set_device(self.args.device)

        self.setup_output_dir()
        self.setup_checkpointer()
        self.setup_logger()

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

    def setup_logger(self):
        """Initialize a text logger to log the experiment.
        
        Each process has its own text logger. The logging message is write to 
        the standard output and a text file named ``worker_n.log`` in the 
        output directory, where ``n`` means the rank of the process. 
        """
        format = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
        formatter = logging.Formatter(fmt=format, datefmt='%Y/%m/%d %H:%M:%S')

        logger.setLevel("INFO")

        # global logger
        stdout = True
        save_path = ""
        logging.basicConfig(
            level=logging.DEBUG if stdout else logging.INFO,
            format=format,
            datefmt='%Y/%m/%d %H:%M:%S',
            filename=save_path if not stdout else None)
        self.logger = logger
