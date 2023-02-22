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
from paddle import nn
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddle.optimizer import AdamW
from paddle.optimizer.lr import StepDecay
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.am_batch_fn import diffsinger_multi_spk_batch_fn
from paddlespeech.t2s.datasets.am_batch_fn import diffsinger_single_spk_batch_fn
from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.diffsinger import DiffSinger
from paddlespeech.t2s.models.diffsinger import DiffSingerEvaluator
from paddlespeech.t2s.models.diffsinger import DiffSingerUpdater
from paddlespeech.t2s.models.diffsinger import DiffusionLoss
from paddlespeech.t2s.models.diffsinger.fastspeech2midi import FastSpeech2MIDILoss
from paddlespeech.t2s.training.extensions.snapshot import Snapshot
from paddlespeech.t2s.training.extensions.visualizer import VisualDL
from paddlespeech.t2s.training.optimizer import build_optimizers
from paddlespeech.t2s.training.seeding import seed_everything
from paddlespeech.t2s.training.trainer import Trainer


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
        "pitch", "energy", "note", "note_dur", "is_slur"
    ]
    converters = {"speech": np.load, "pitch": np.load, "energy": np.load}
    spk_num = None
    if args.speaker_dict is not None:
        print("multiple speaker diffsinger!")
        collate_fn = diffsinger_multi_spk_batch_fn
        with open(args.speaker_dict, 'rt') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
        fields += ["spk_id"]
    else:
        collate_fn = diffsinger_single_spk_batch_fn
        print("single speaker diffsinger!")

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
    config["model"]["fastspeech2_params"]["spk_num"] = spk_num
    model = DiffSinger(idim=vocab_size, odim=odim, **config["model"])
    model_fs2 = model.fs2
    model_ds = model.diffusion
    if world_size > 1:
        model = DataParallel(model)
        model_fs2 = model._layers.fs2
        model_ds = model._layers.diffusion
    print("models done!")

    criterion_fs2 = FastSpeech2MIDILoss(**config["fs2_updater"])
    criterion_ds = DiffusionLoss(**config["ds_updater"])
    print("criterions done!")

    optimizer_fs2 = build_optimizers(model_fs2, **config["fs2_optimizer"])
    gradient_clip_ds = nn.ClipGradByGlobalNorm(config["ds_grad_norm"])
    optimizer_ds = AdamW(
        learning_rate=config["ds_scheduler_params"]["learning_rate"],
        grad_clip=gradient_clip_ds,
        parameters=model_ds.parameters(),
        **config["ds_optimizer_params"])
    print("optimizer done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        config_name = args.config.split("/")[-1]
        # copy conf to output_dir
        shutil.copyfile(args.config, output_dir / config_name)

    updater = DiffSingerUpdater(
        model=model,
        optimizers={
            "fs2": optimizer_fs2,
            "ds": optimizer_ds,
        },
        criterions={
            "fs2": criterion_fs2,
            "ds": criterion_ds,
        },
        dataloader=train_dataloader,
        ds_train_start_steps=config.ds_train_start_steps,
        output_dir=output_dir)

    evaluator = DiffSingerEvaluator(
        model=model,
        criterions={
            "fs2": criterion_fs2,
            "ds": criterion_ds,
        },
        dataloader=dev_dataloader,
        output_dir=output_dir, )

    trainer = Trainer(
        updater,
        stop_trigger=(config.train_max_steps, "iteration"),
        out=output_dir, )

    if dist.get_rank() == 0:
        trainer.extend(
            evaluator, trigger=(config.eval_interval_steps, 'iteration'))
        trainer.extend(VisualDL(output_dir), trigger=(1, 'iteration'))
    trainer.extend(
        Snapshot(max_size=config.num_snapshots),
        trigger=(config.save_interval_steps, 'iteration'))

    print("Trainer Done!")
    trainer.run()


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Train a DiffSinger model.")
    parser.add_argument("--config", type=str, help="diffsinger config file.")
    parser.add_argument("--train-metadata", type=str, help="training data.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu=0, use cpu.")
    parser.add_argument(
        "--phones-dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--speaker-dict",
        type=str,
        default=None,
        help="speaker id map file for multiple speaker model.")

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
