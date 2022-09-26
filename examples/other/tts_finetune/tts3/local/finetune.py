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
from typing import List

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


class TrainArgs():
    def __init__(self,
                 ngpu,
                 config_file,
                 dump_dir: Path,
                 output_dir: Path,
                 frozen_layers: List[str]):
        # config: fastspeech2 config file.
        self.config = str(config_file)
        self.train_metadata = str(dump_dir / "train/norm/metadata.jsonl")
        self.dev_metadata = str(dump_dir / "dev/norm/metadata.jsonl")
        # model output dir.
        self.output_dir = str(output_dir)
        self.ngpu = ngpu
        self.phones_dict = str(dump_dir / "phone_id_map.txt")
        self.speaker_dict = str(dump_dir / "speaker_id_map.txt")
        self.voice_cloning = False
        # frozen layers
        self.frozen_layers = frozen_layers


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
    if args.frozen_layers != []:
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


if __name__ == '__main__':
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="./pretrained_models/fastspeech2_aishell3_ckpt_1.1.0",
        help="Path to pretrained model")

    parser.add_argument(
        "--dump_dir",
        type=str,
        default="./dump",
        help="directory to save feature files and metadata.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exp/default/",
        help="directory to save finetune model.")

    parser.add_argument(
        "--ngpu", type=int, default=2, help="if ngpu=0, use cpu.")

    parser.add_argument("--epoch", type=int, default=100, help="finetune epoch")
    parser.add_argument(
        "--finetune_config",
        type=str,
        default="./finetune.yaml",
        help="Path to finetune config file")

    args = parser.parse_args()

    dump_dir = Path(args.dump_dir).expanduser()
    dump_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrained_model_dir = Path(args.pretrained_model_dir).expanduser()

    # read config
    config_file = pretrained_model_dir / "default.yaml"
    with open(config_file) as f:
        config = CfgNode(yaml.safe_load(f))
    config.max_epoch = config.max_epoch + args.epoch

    with open(args.finetune_config) as f2:
        finetune_config = CfgNode(yaml.safe_load(f2))
    config.batch_size = finetune_config.batch_size if finetune_config.batch_size > 0 else config.batch_size
    config.optimizer.learning_rate = finetune_config.learning_rate if finetune_config.learning_rate > 0 else config.optimizer.learning_rate
    config.num_snapshots = finetune_config.num_snapshots if finetune_config.num_snapshots > 0 else config.num_snapshots
    frozen_layers = finetune_config.frozen_layers
    assert type(frozen_layers) == list, "frozen_layers should be set a list."

    # create a new args for training
    train_args = TrainArgs(args.ngpu, config_file, dump_dir, output_dir,
                           frozen_layers)

    # finetune models
    # dispatch
    if args.ngpu > 1:
        dist.spawn(train_sp, (train_args, config), nprocs=args.ngpu)
    else:
        train_sp(train_args, config)
