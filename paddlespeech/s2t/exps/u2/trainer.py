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
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader

from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.io.dataset import ManifestDataset
from paddlespeech.s2t.io.sampler import SortagradBatchSampler
from paddlespeech.s2t.io.sampler import SortagradDistributedBatchSampler
from paddlespeech.s2t.models.u2 import U2Evaluator
from paddlespeech.s2t.models.u2 import U2Model
from paddlespeech.s2t.models.u2 import U2Updater
from paddlespeech.s2t.training.extensions.snapshot import Snapshot
from paddlespeech.s2t.training.extensions.visualizer import VisualDL
from paddlespeech.s2t.training.optimizer import OptimizerFactory
from paddlespeech.s2t.training.scheduler import LRSchedulerFactory
from paddlespeech.s2t.training.timer import Timer
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.training.updaters.trainer import Trainer as NewTrainer
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig

logger = Log(__name__).getlog()


class U2Trainer(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)

    def setup_dataloader(self):
        config = self.config.clone()
        config.defrost()
        config.keep_transcription_text = False

        # train/valid dataset, return token ids
        config.manifest = config.train_manifest
        train_dataset = ManifestDataset.from_config(config)

        config.manifest = config.dev_manifest
        dev_dataset = ManifestDataset.from_config(config)

        collate_fn_train = SpeechCollator.from_config(config)

        collate_fn_dev = SpeechCollator.from_config(config)

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
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn_train,
            num_workers=config.num_workers,
        )
        self.valid_loader = DataLoader(
            dev_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn_dev,
            num_workers=config.num_workers,
        )

        # test dataset, return raw text
        config.manifest = config.test_manifest
        # filter test examples, will cause less examples, but no mismatch with training
        # and can use large batch size , save training time, so filter test egs now.
        config.min_input_len = 0.0  # second
        config.max_input_len = float('inf')  # second
        config.min_output_len = 0.0  # tokens
        config.max_output_len = float('inf')  # tokens
        config.min_output_input_ratio = 0.00
        config.max_output_input_ratio = float('inf')

        test_dataset = ManifestDataset.from_config(config)
        # return text ord id
        config.keep_transcription_text = True
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.decode.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=SpeechCollator.from_config(config))
        # return text token id
        config.keep_transcription_text = False
        self.align_loader = DataLoader(
            test_dataset,
            batch_size=config.decode.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=SpeechCollator.from_config(config))
        logger.info("Setup train/valid/test/align Dataloader!")

    def setup_model(self):
        config = self.config
        model_conf = config
        with UpdateConfig(model_conf):
            model_conf.input_dim = self.train_loader.collate_fn.feature_size
            model_conf.output_dim = self.train_loader.collate_fn.vocab_size

        model = U2Model.from_config(model_conf)

        if self.parallel:
            model = paddle.DataParallel(model)

        model.train()
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
            lr_scheduler=None,
        ):
            train_config = config
            optim_type = train_config.optim
            optim_conf = train_config.optim_conf
            scheduler_type = train_config.scheduler
            scheduler_conf = train_config.scheduler_conf
            return {
                "grad_clip": train_config.global_grad_clip,
                "weight_decay": optim_conf.weight_decay,
                "learning_rate":
                lr_scheduler if lr_scheduler else optim_conf.lr,
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

    def setup_updater(self):
        output_dir = self.output_dir
        config = self.config

        updater = U2Updater(model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.lr_scheduler,
                            dataloader=self.train_loader,
                            output_dir=output_dir,
                            accum_grad=config.accum_grad)

        trainer = NewTrainer(updater, (config.n_epoch, 'epoch'), output_dir)

        evaluator = U2Evaluator(self.model, self.valid_loader)

        trainer.extend(evaluator, trigger=(1, "epoch"))

        if dist.get_rank() == 0:
            trainer.extend(VisualDL(output_dir), trigger=(1, "iteration"))
            num_snapshots = config.checkpoint.kbest_n
            trainer.extend(Snapshot(mode='kbest',
                                    max_size=num_snapshots,
                                    indicator='VALID/LOSS',
                                    less_better=True),
                           trigger=(1, 'epoch'))
        # print(trainer.extensions)
        # trainer.run()
        self.trainer = trainer

    def run(self):
        """The routine of the experiment after setup. This method is intended
        to be used by the user.
        """
        self.setup_updater()
        with Timer("Training Done: {}"):
            self.trainer.run()
