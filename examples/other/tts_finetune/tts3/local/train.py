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
import argparse
import logging
import os
import shutil
from pathlib import Path

import jsonlines
import numpy as np
import paddle
import yaml
from paddle import DataParallel
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.am_batch_fn import fastspeech2_multi_spk_batch_fn
from paddlespeech.t2s.datasets.am_batch_fn import fastspeech2_single_spk_batch_fn
from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Evaluator
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Updater
from paddlespeech.t2s.training.extensions.snapshot import Snapshot
from paddlespeech.t2s.training.extensions.visualizer import VisualDL
from paddlespeech.t2s.training.optimizer import build_optimizers
from paddlespeech.t2s.training.seeding import seed_everything
from paddlespeech.t2s.training.trainer import Trainer
from paddlespeech.t2s.utils import str2bool
from typing import List


def freeze_layer(model, layers: List[str]):
    """freeze layers

    Args:
        layers (List[str]): frozen layers
    """
    for layer in layers:
        for param in eval("model." + layer + ".parameters()"):
            param.trainable = False


def train_sp(args, config):
    # decides device type and whether to run in parallel
    # setup running environment correctly
    if (not paddle.is_compiled_with_cuda()) or args.ngpu == 0:
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu")
    world_size = paddle.distributed.get_world_size()
    if world_size > 1:
        paddle.distributed.init_parallel_env()

    # set the random seed, it is a must for multiprocess training
    seed_everything(config.seed)

    print(
        f"rank: {dist.get_rank()}, pid: {os.getpid()}, parent_pid: {os.getppid()}",
    )
    fields = [
        "text", "text_lengths", "speech", "speech_lengths", "durations",
        "pitch", "energy"
    ]
    converters = {"speech": np.load, "pitch": np.load, "energy": np.load}
    spk_num = None
    if args.speaker_dict is not None:
        print("multiple speaker fastspeech2!")
        collate_fn = fastspeech2_multi_spk_batch_fn
        with open(args.speaker_dict, 'rt') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
        fields += ["spk_id"]
    elif args.voice_cloning:
        print("Training voice cloning!")
        collate_fn = fastspeech2_multi_spk_batch_fn
        fields += ["spk_emb"]
        converters["spk_emb"] = np.load
    else:
        print("single speaker fastspeech2!")
        collate_fn = fastspeech2_single_spk_batch_fn
    print("spk_num:", spk_num)

    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for training and validation
    with jsonlines.open(args.train_metadata, 'r') as reader:
        train_metadata = list(reader)
    train_dataset = DataTable(
        data=train_metadata,
        fields=fields,
        converters=converters, )
    with jsonlines.open(args.dev_metadata, 'r') as reader:
        dev_metadata = list(reader)
    dev_dataset = DataTable(
        data=dev_metadata,
        fields=fields,
        converters=converters, )

    # collate function and dataloader

    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True)

    print("samplers done!")

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers)

    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers)
    print("dataloaders done!")

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    odim = config.n_mels
    model = FastSpeech2(
        idim=vocab_size, odim=odim, spk_num=spk_num, **config["model"])

    # freeze layer
    if args.frozen_layers is not None:
        freeze_layer(model, args.frozen_layers)

    if world_size > 1:
        model = DataParallel(model)
    print("model done!")

    optimizer = build_optimizers(model, **config["optimizer"])
    print("optimizer done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        config_name = args.config.split("/")[-1]
        # copy conf to output_dir
        shutil.copyfile(args.config, output_dir / config_name)

    updater = FastSpeech2Updater(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        output_dir=output_dir,
        **config["updater"])

    trainer = Trainer(updater, (config.max_epoch, 'epoch'), output_dir)

    evaluator = FastSpeech2Evaluator(
        model, dev_dataloader, output_dir=output_dir, **config["updater"])

    if dist.get_rank() == 0:
        trainer.extend(evaluator, trigger=(1, "epoch"))
        trainer.extend(VisualDL(output_dir), trigger=(1, "iteration"))
    trainer.extend(
        Snapshot(max_size=config.num_snapshots), trigger=(1, 'epoch'))
    trainer.run()
