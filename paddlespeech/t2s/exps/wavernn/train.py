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
from paddle.optimizer import Adam
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.datasets.vocoder_batch_fn import WaveRNNClip
from paddlespeech.t2s.models.wavernn import WaveRNN
from paddlespeech.t2s.models.wavernn import WaveRNNEvaluator
from paddlespeech.t2s.models.wavernn import WaveRNNUpdater
from paddlespeech.t2s.modules.losses import discretized_mix_logistic_loss
from paddlespeech.t2s.training.extensions.snapshot import Snapshot
from paddlespeech.t2s.training.extensions.visualizer import VisualDL
from paddlespeech.t2s.training.seeding import seed_everything
from paddlespeech.t2s.training.trainer import Trainer


def train_sp(args, config):
    # decides device type and whether to run in parallel
    # setup running environment correctly
    world_size = paddle.distributed.get_world_size()
    if (not paddle.is_compiled_with_cuda()) or args.ngpu == 0:
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

    # construct dataset for training and validation
    with jsonlines.open(args.train_metadata, 'r') as reader:
        train_metadata = list(reader)
    train_dataset = DataTable(
        data=train_metadata,
        fields=["wave", "feats"],
        converters={
            "wave": np.load,
            "feats": np.load,
        },
    )

    with jsonlines.open(args.dev_metadata, 'r') as reader:
        dev_metadata = list(reader)
    dev_dataset = DataTable(
        data=dev_metadata,
        fields=["wave", "feats"],
        converters={
            "wave": np.load,
            "feats": np.load,
        },
    )

    batch_fn = WaveRNNClip(mode=config.model.mode,
                           aux_context_window=config.model.aux_context_window,
                           hop_size=config.n_shift,
                           batch_max_steps=config.batch_max_steps,
                           bits=config.model.bits)

    # collate function and dataloader
    train_sampler = DistributedBatchSampler(train_dataset,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            drop_last=True)
    dev_sampler = DistributedBatchSampler(dev_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          drop_last=False)
    print("samplers done!")

    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  collate_fn=batch_fn,
                                  num_workers=config.num_workers)

    dev_dataloader = DataLoader(dev_dataset,
                                collate_fn=batch_fn,
                                batch_sampler=dev_sampler,
                                num_workers=config.num_workers)

    valid_generate_loader = DataLoader(dev_dataset, batch_size=1)

    print("dataloaders done!")

    model = WaveRNN(hop_length=config.n_shift,
                    sample_rate=config.fs,
                    **config["model"])
    if world_size > 1:
        model = DataParallel(model)
    print("model done!")

    if config.model.mode == 'RAW':
        criterion = paddle.nn.CrossEntropyLoss(axis=1)
    elif config.model.mode == 'MOL':
        criterion = discretized_mix_logistic_loss
    else:
        criterion = None
        RuntimeError('Unknown model mode value - ', config.model.mode)
    print("criterions done!")
    clip = paddle.nn.ClipGradByGlobalNorm(config.grad_clip)
    optimizer = Adam(parameters=model.parameters(),
                     learning_rate=config.learning_rate,
                     grad_clip=clip)

    print("optimizer done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        config_name = args.config.split("/")[-1]
        # copy conf to output_dir
        shutil.copyfile(args.config, output_dir / config_name)

    updater = WaveRNNUpdater(model=model,
                             optimizer=optimizer,
                             criterion=criterion,
                             dataloader=train_dataloader,
                             output_dir=output_dir,
                             mode=config.model.mode)

    evaluator = WaveRNNEvaluator(model=model,
                                 dataloader=dev_dataloader,
                                 criterion=criterion,
                                 output_dir=output_dir,
                                 valid_generate_loader=valid_generate_loader,
                                 config=config)

    trainer = Trainer(updater,
                      stop_trigger=(config.train_max_steps, "iteration"),
                      out=output_dir)

    if dist.get_rank() == 0:
        trainer.extend(evaluator,
                       trigger=(config.eval_interval_steps, 'iteration'))
        trainer.extend(VisualDL(output_dir), trigger=(1, 'iteration'))
    trainer.extend(Snapshot(max_size=config.num_snapshots),
                   trigger=(config.save_interval_steps, 'iteration'))

    print("Trainer Done!")
    trainer.run()


def main():
    # parse args and config and redirect to train_sp

    parser = argparse.ArgumentParser(description="Train a WaveRNN model.")
    parser.add_argument("--config", type=str, help="WaveRNN config file.")
    parser.add_argument("--train-metadata", type=str, help="training data.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="if ngpu == 0, use cpu.")

    args = parser.parse_args()

    with open(args.config, 'rt') as f:
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
