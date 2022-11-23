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

import paddle
import yaml
from paddle import DataParallel
from paddle import distributed as dist
from paddle import nn
from paddle.io import DataLoader
from paddle.optimizer import Adam
from paddle.optimizer.lr import ExponentialDecay
from yacs.config import CfgNode

from paddlespeech.t2s.training.extensions.snapshot import Snapshot
from paddlespeech.t2s.training.extensions.visualizer import VisualDL
from paddlespeech.t2s.training.seeding import seed_everything
from paddlespeech.t2s.training.trainer import Trainer
from paddlespeech.text.models.ernie_linear import ErnieLinear
from paddlespeech.text.models.ernie_linear import ErnieLinearEvaluator
from paddlespeech.text.models.ernie_linear import ErnieLinearUpdater
from paddlespeech.text.models.ernie_linear import PuncDataset
from paddlespeech.text.models.ernie_linear import PuncDatasetFromErnieTokenizer

DefinedClassifier = {
    'ErnieLinear': ErnieLinear,
}

DefinedLoss = {
    "ce": nn.CrossEntropyLoss,
}

DefinedDataset = {
    'Punc': PuncDataset,
    'Ernie': PuncDatasetFromErnieTokenizer,
}


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
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True
    train_dataset = DefinedDataset[config["dataset_type"]](
        train_path=config["train_path"], **config["data_params"])
    dev_dataset = DefinedDataset[config["dataset_type"]](
        train_path=config["dev_path"], **config["data_params"])
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=config.num_workers,
        batch_size=config.batch_size)

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers)

    print("dataloaders done!")

    model = DefinedClassifier[config["model_type"]](**config["model"])

    if world_size > 1:
        model = DataParallel(model)
    print("model done!")

    criterion = DefinedLoss[config["loss_type"]](
        **config["loss"]) if "loss_type" in config else DefinedLoss["ce"]()

    print("criterions done!")

    lr_schedule = ExponentialDecay(**config["scheduler_params"])
    optimizer = Adam(
        learning_rate=lr_schedule,
        parameters=model.parameters(),
        weight_decay=paddle.regularizer.L2Decay(
            config["optimizer_params"]["weight_decay"]))

    print("optimizer done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        config_name = args.config.split("/")[-1]
        # copy conf to output_dir
        shutil.copyfile(args.config, output_dir / config_name)

    updater = ErnieLinearUpdater(
        model=model,
        criterion=criterion,
        scheduler=lr_schedule,
        optimizer=optimizer,
        dataloader=train_dataloader,
        output_dir=output_dir)

    trainer = Trainer(updater, (config.max_epoch, 'epoch'), output_dir)

    evaluator = ErnieLinearEvaluator(
        model=model,
        criterion=criterion,
        dataloader=dev_dataloader,
        output_dir=output_dir)

    if dist.get_rank() == 0:
        trainer.extend(evaluator, trigger=(1, "epoch"))
        trainer.extend(VisualDL(output_dir), trigger=(1, "iteration"))
    trainer.extend(
        Snapshot(max_size=config.num_snapshots), trigger=(1, 'epoch'))
    trainer.run()


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Train a ErnieLinear model.")
    parser.add_argument("--config", type=str, help="ErnieLinear config file.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu=0, use cpu.")

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
