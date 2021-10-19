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

import jsonlines
import numpy as np
import paddle
import yaml
from paddle import distributed as dist
from paddle import DataParallel
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from parakeet.datasets.data_table import DataTable
from parakeet.datasets.am_batch_fn import speedyspeech_batch_fn
from parakeet.models.speedyspeech import SpeedySpeech
from parakeet.models.speedyspeech import SpeedySpeechEvaluator
from parakeet.models.speedyspeech import SpeedySpeechUpdater
from parakeet.training.extensions.snapshot import Snapshot
from parakeet.training.extensions.visualizer import VisualDL
from parakeet.training.optimizer import build_optimizers
from parakeet.training.seeding import seed_everything
from parakeet.training.trainer import Trainer
from pathlib import Path
from visualdl import LogWriter
from yacs.config import CfgNode


def train_sp(args, config):
    # decides device type and whether to run in parallel
    # setup running environment correctly
    world_size = paddle.distributed.get_world_size()
    if not paddle.is_compiled_with_cuda():
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu")
        if world_size > 1:
            paddle.distributed.init_parallel_env()

    # set the random seed, it is a must for multiprocess training
    seed_everything(config.seed)

    print(
        f"rank: {dist.get_rank()}, pid: {os.getpid()}, parent_pid: {os.getppid()}",
    )

    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for training and validation
    with jsonlines.open(args.train_metadata, 'r') as reader:
        train_metadata = list(reader)
    if args.use_relative_path:
        # if use_relative_path in preprocess, covert it to absolute path here
        metadata_dir = Path(args.train_metadata).parent
        for item in train_metadata:
            item["feats"] = str(metadata_dir / item["feats"])

    train_dataset = DataTable(
        data=train_metadata,
        fields=[
            "phones", "tones", "num_phones", "num_frames", "feats", "durations"
        ],
        converters={
            "feats": np.load,
        }, )
    with jsonlines.open(args.dev_metadata, 'r') as reader:
        dev_metadata = list(reader)
    if args.use_relative_path:
        # if use_relative_path in preprocess, covert it to absolute path here
        metadata_dir = Path(args.dev_metadata).parent
        for item in dev_metadata:
            item["feats"] = str(metadata_dir / item["feats"])

    dev_dataset = DataTable(
        data=dev_metadata,
        fields=[
            "phones", "tones", "num_phones", "num_frames", "feats", "durations"
        ],
        converters={
            "feats": np.load,
        }, )

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
        collate_fn=speedyspeech_batch_fn,
        num_workers=config.num_workers)
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=config.batch_size,
        collate_fn=speedyspeech_batch_fn,
        num_workers=config.num_workers)
    print("dataloaders done!")
    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)
    with open(args.tones_dict, "r") as f:
        tone_id = [line.strip().split() for line in f.readlines()]
    tone_size = len(tone_id)
    print("tone_size:", tone_size)

    model = SpeedySpeech(
        vocab_size=vocab_size, tone_size=tone_size, **config["model"])
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

    updater = SpeedySpeechUpdater(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        output_dir=output_dir)

    trainer = Trainer(updater, (config.max_epoch, 'epoch'), output_dir)

    evaluator = SpeedySpeechEvaluator(
        model, dev_dataloader, output_dir=output_dir)

    if dist.get_rank() == 0:
        trainer.extend(evaluator, trigger=(1, "epoch"))
        writer = LogWriter(str(output_dir))
        trainer.extend(VisualDL(writer), trigger=(1, "iteration"))
        trainer.extend(
            Snapshot(max_size=config.num_snapshots), trigger=(1, 'epoch'))
    trainer.run()


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Train a Speedyspeech model with sigle speaker dataset.")
    parser.add_argument("--config", type=str, help="config file.")
    parser.add_argument("--train-metadata", type=str, help="training data.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--device", type=str, default="gpu", help="device type to use.")
    parser.add_argument(
        "--nprocs", type=int, default=1, help="number of processes.")
    parser.add_argument("--verbose", type=int, default=1, help="verbose.")

    def str2bool(str):
        return True if str.lower() == 'true' else False

    parser.add_argument(
        "--use-relative-path",
        type=str2bool,
        default=False,
        help="whether use relative path in metadata")

    parser.add_argument(
        "--phones-dict", type=str, default=None, help="phone vocabulary file.")

    parser.add_argument(
        "--tones-dict", type=str, default=None, help="tone vocabulary file.")

    # 这里可以多传入 max_epoch 等
    args, rest = parser.parse_known_args()
    if args.device == "cpu" and args.nprocs > 1:
        raise RuntimeError("Multiprocess training on CPU is not supported.")
    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))

    if rest:
        extra = []
        # to support key=value format
        for item in rest:
            # remove "--"
            item = item[2:]
            extra.extend(item.split("=", maxsplit=1))
        config.merge_from_list(extra)

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)
    print(
        f"master see the word size: {dist.get_world_size()}, from pid: {os.getpid()}"
    )

    # dispatch
    if args.nprocs > 1:
        dist.spawn(train_sp, (args, config), nprocs=args.nprocs)
    else:
        train_sp(args, config)


if __name__ == "__main__":
    main()
