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

from paddlespeech.t2s.datasets.am_batch_fn import transformer_single_spk_batch_fn
from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.transformer_tts import TransformerTTS
from paddlespeech.t2s.models.transformer_tts import TransformerTTSEvaluator
from paddlespeech.t2s.models.transformer_tts import TransformerTTSUpdater
from paddlespeech.t2s.training.extensions.snapshot import Snapshot
from paddlespeech.t2s.training.extensions.visualizer import VisualDL
from paddlespeech.t2s.training.optimizer import build_optimizers
from paddlespeech.t2s.training.seeding import seed_everything
from paddlespeech.t2s.training.trainer import Trainer


def train_sp(args, config):
    # decides device type and whether to run in parallel
    # setup running environment correctly
    if paddle.is_compiled_with_cuda() and args.ngpu > 0:
        paddle.set_device("gpu")
    elif paddle.is_compiled_with_npu() and args.ngpu > 0:
        paddle.set_device("npu")
    else:
        paddle.set_device("cpu")
    world_size = paddle.distributed.get_world_size()
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
    train_dataset = DataTable(
        data=train_metadata,
        fields=[
            "text",
            "text_lengths",
            "speech",
            "speech_lengths",
        ],
        converters={
            "speech": np.load,
        }, )
    with jsonlines.open(args.dev_metadata, 'r') as reader:
        dev_metadata = list(reader)
    dev_dataset = DataTable(
        data=dev_metadata,
        fields=[
            "text",
            "text_lengths",
            "speech",
            "speech_lengths",
        ],
        converters={
            "speech": np.load,
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
        collate_fn=transformer_single_spk_batch_fn,
        num_workers=config.num_workers)

    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=config.batch_size,
        collate_fn=transformer_single_spk_batch_fn,
        num_workers=config.num_workers)
    print("dataloaders done!")

    with open(args.phones_dict, 'r', encoding='utf-8') as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    odim = config.n_mels
    model = TransformerTTS(idim=vocab_size, odim=odim, **config["model"])
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

    updater = TransformerTTSUpdater(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        output_dir=output_dir,
        **config["updater"])

    trainer = Trainer(updater, (config.max_epoch, 'epoch'), output_dir)

    evaluator = TransformerTTSEvaluator(
        model, dev_dataloader, output_dir=output_dir, **config["updater"])

    if dist.get_rank() == 0:
        trainer.extend(evaluator, trigger=(1, "epoch"))
        trainer.extend(VisualDL(output_dir), trigger=(1, "iteration"))
    trainer.extend(
        Snapshot(max_size=config.num_snapshots), trigger=(1, 'epoch'))
    trainer.run()


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Train a TransformerTTS "
                                     "model with LJSpeech TTS dataset.")
    parser.add_argument(
        "--config", type=str, help="TransformerTTS config file.")
    parser.add_argument("--train-metadata", type=str, help="training data.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument(
        "--phones-dict", type=str, default=None, help="phone vocabulary file.")

    args = parser.parse_args()

    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)
    print(
        f"master see the word size: {dist.get_world_size()}, from pid: {os.getpid()}"
    )

    # dispatch
    if args.ngpu > 1:
        dist.spawn(train_sp, (args, config), nprocs=args.ngpu)
    else:
        train_sp(args, config)


if __name__ == "__main__":
    main()
