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
from paddle.optimizer import Adam
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.am_batch_fn import build_erniesat_collate_fn
from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.ernie_sat import ErnieSAT
from paddlespeech.t2s.models.ernie_sat import ErnieSATEvaluator
from paddlespeech.t2s.models.ernie_sat import ErnieSATUpdater
from paddlespeech.t2s.training.extensions.snapshot import Snapshot
from paddlespeech.t2s.training.extensions.visualizer import VisualDL
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
        "text", "text_lengths", "speech", "speech_lengths", "align_start",
        "align_end"
    ]
    converters = {"speech": np.load}
    spk_num = None

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
    collate_fn = build_erniesat_collate_fn(
        mlm_prob=config.mlm_prob,
        mean_phn_span=config.mean_phn_span,
        seg_emb=config.model['enc_input_layer'] == 'sega_mlm',
        text_masking=config["model"]["text_masking"],
        epoch=config["max_epoch"])

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
    model = ErnieSAT(idim=vocab_size, odim=odim, **config["model"])
    # model_path = "/home/yuantian01/PaddleSpeech_ERNIE_SAT/PaddleSpeech/examples/ernie_sat/pretrained_model/paddle_checkpoint_en/model.pdparams"
    # state_dict = paddle.load(model_path)
    # new_state_dict = {}
    # for key, value in state_dict.items():
    #     new_key = "model." + key
    #     new_state_dict[new_key] = value
    # model.set_state_dict(new_state_dict)

    if world_size > 1:
        model = DataParallel(model)
    print("model done!")

    scheduler = paddle.optimizer.lr.NoamDecay(
        d_model=config["scheduler_params"]["d_model"],
        warmup_steps=config["scheduler_params"]["warmup_steps"])
    grad_clip = nn.ClipGradByGlobalNorm(config["grad_clip"])
    optimizer = Adam(
        learning_rate=scheduler,
        grad_clip=grad_clip,
        parameters=model.parameters())

    print("optimizer done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        config_name = args.config.split("/")[-1]
        # copy conf to output_dir
        shutil.copyfile(args.config, output_dir / config_name)

    updater = ErnieSATUpdater(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_dataloader,
        text_masking=config["model"]["text_masking"],
        odim=odim,
        output_dir=output_dir)

    trainer = Trainer(updater, (config.max_epoch, 'epoch'), output_dir)

    evaluator = ErnieSATEvaluator(
        model=model,
        dataloader=dev_dataloader,
        text_masking=config["model"]["text_masking"],
        odim=odim,
        output_dir=output_dir, )

    if dist.get_rank() == 0:
        trainer.extend(evaluator, trigger=(1, "epoch"))
        trainer.extend(VisualDL(output_dir), trigger=(1, "iteration"))
    trainer.extend(
        Snapshot(max_size=config.num_snapshots), trigger=(1, 'epoch'))
    trainer.run()


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Train an ErnieSAT model.")
    parser.add_argument("--config", type=str, help="ErnieSAT config file.")
    parser.add_argument("--train-metadata", type=str, help="training data.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu=0, use cpu.")
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
