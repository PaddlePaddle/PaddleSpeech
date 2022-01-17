#!/usr/bin/python3
#! coding:utf-8

import time

from paddlespeech.s2t.io.dataset import ManifestDataset
from paddlespeech.s2t.io.dataset import CSVManifestDataset
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.io.collator import SIDSpeechCollator
from paddlespeech.s2t.io.sampler import SortagradBatchSampler
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils.log import Log 
from paddlespeech.s2t.training.optimizer import OptimizerFactory
from paddle.io import DataLoader
from paddlespeech.vector.models.ecapa_tdnn import EcapaTdnn
from paddlespeech.vector.models.ecapa_tdnn import CosClassifier
from paddlespeech.vector.models.ecapa_tdnn import AdditiveAngularMargin
from paddlespeech.vector.models.ecapa_tdnn import LogSoftmaxWrapper
from paddlespeech.s2t.training.scheduler import LRSchedulerFactory
import paddle 

logger = Log(__name__).getlog()

class SIDTrainer(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)

    def setup_dataloader(self):
        config = self.config.clone()
        config.defrost()
        logger.info("config: \n{}".format(config))

        config.manifest = config.train_manifest
        train_dataset = CSVManifestDataset.from_config(config)
        collate_fn_train = SIDSpeechCollator.from_config(config)
        if self.parallel:
            # 多GPU训练的时候，使用分布式数据
            pass
        else:
            logger.info("train dataset: {}".format(train_dataset))
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
                        num_workers=config.num_workers
                        )

    def setup_model(self):
        config = self.config
        # logger.info("sid model config: {}".format(config))
        config.defrost()
        config.input_size = config.feat_dim
        if config["speaker_model"].lower() == "ecapa-xvector":
            self.model = EcapaTdnn.init_from_config(config)
        else:
            logger.error("no such speaker model type: {}".format(config["speaker_model"]))
            exit(-1)

        if config["classifier_type"].lower() == "cosclassifier":
            self.classifier = CosClassifier.init_from_config(config)
        else:
            logger.error("no such classifier model type: {}".format(config["classifier_type"]))
            exit(-1)    

        
        if config["loss_type"].lower() == "logsoftmaxwrapper":
            if config["loss_conf"]["loss_fn"].lower() == "additiveangularmargin":
                margin = config["loss_conf"]["margin"]
                scale = config["loss_conf"]["scale"]
                self.loss_fn = AdditiveAngularMargin(margin=margin, scale=scale)
            else:
                logger.error("no such loss pre-function : {}".format(config["loss_conf"]["loss_fn"]))
                exit(-1)                 
            self.loss = LogSoftmaxWrapper(self.loss_fn)
        else:
            logger.error("no such classifier model type: {}".format(config["classifier_type"]))
            exit(-1)                
        # logger.info("sid model config type: {}".format(type(config)))
        # logger.info("model type: {}".format(type(self.model)))

        if config["optim"].lower() == "adam":
            optim_type = config.optim
            optim_conf = config.optim_conf
            scheduler_type = config.scheduler
            scheduler_conf = config.scheduler_conf
            scheduler_args = {
                "learning_rate": optim_conf.lr,
                "verbose": False,
                "warmup_steps": scheduler_conf.warmup_steps,
                "gamma": scheduler_conf.lr_decay,
                "d_model": config.model_conf.attention_channels,
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
            optimzer_args = optimizer_args(config, self.model.parameters(), lr_scheduler)
            self.optimizer = OptimizerFactory.from_args(optim_type, optimzer_args)
            self.lr_scheduler = lr_scheduler
        else:
            logger.error("no such optimizer model type: {}".format(config["opt_type"]))
            exit(-1)      
                   
    def do_train(self):
        # self.before_train()
        logger.info("train total utterances examples: {}".format(len(self.train_loader.dataset)))
        logger.info("epoch: {}".format(self.epoch))
        logger.info("total epoch: {}".format(self.config.n_epoch))

        while self.epoch < self.config.n_epoch:
            logger.info("cur epoch: {}".format(self.epoch))
            data_start_time = time.time()
            for step, batch in enumerate(self.train_loader):
                dataload_time = time.time() - data_start_time
                msg = "data load time: {:>.3f}".format(dataload_time)
                logger.info(msg)
                data_start_time = time.time()

                # 开始计算模型的结果
                # xs_pad = [batch, time, dim]
                utts, xs_pad, ilens, spk_ids = batch
                logger.info("xs_pad shape: {}".format(xs_pad.shape))
                logger.info("spk_ids shape: {}".format(spk_ids.shape))
                
                # 经过transpose之后，xs_pad = [batch, dim, time]
                xs_pad = paddle.transpose(xs_pad, perm=[0,2,1])
                model_output = self.model(xs_pad)
                logits = self.classifier(model_output)
                
                # 计算 loss 
                loss = self.loss(logits, spk_ids)

                batch_train_time = time.time() - data_start_time
                data_start_time = time.time()

                self.optimizer.step()
                self.optimizer.clear_grad()
                self.lr_scheduler.step()

            # 开始新的epoch
            self.epoch += 1

    def run(self):
        logger.info("start to training")
        self.do_train()


        