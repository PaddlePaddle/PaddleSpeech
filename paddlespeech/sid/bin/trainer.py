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
from paddle.io import DataLoader

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

                utts, xs_pad, ilens, spk_ids = batch
                logger.info("xs_pad shape: {}".format(xs_pad.shape))
                logger.info("spk_ids shape: {}".format(spk_ids.shape))

                batch_train_time = time.time() - data_start_time
                data_start_time = time.time()

            # 开始新的epoch
            self.epoch += 1

    def run(self):
        logger.info("start to training")
        self.do_train()


        