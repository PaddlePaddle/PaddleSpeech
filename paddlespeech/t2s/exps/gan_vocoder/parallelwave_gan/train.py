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
from paddle.optimizer import Adam  # No RAdaom
from paddle.optimizer.lr import StepDecay
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.datasets.vocoder_batch_fn import Clip
from paddlespeech.t2s.models.parallel_wavegan import PWGDiscriminator
from paddlespeech.t2s.models.parallel_wavegan import PWGEvaluator
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator
from paddlespeech.t2s.models.parallel_wavegan import PWGUpdater
from paddlespeech.t2s.modules.losses import MultiResolutionSTFTLoss
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

    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for training and validation
    with jsonlines.open(args.train_metadata, 'r') as reader:
        train_metadata = list(reader)
    train_dataset = DataTable(
        data=train_metadata,
        fields=["wave", "feats"],
        converters={
            "wave": np.load,
            "feats": np.load,
        }, )
    with jsonlines.open(args.dev_metadata, 'r') as reader:
        dev_metadata = list(reader)
    dev_dataset = DataTable(
        data=dev_metadata,
        fields=["wave", "feats"],
        converters={
            "wave": np.load,
            "feats": np.load,
        }, )

    # collate function and dataloader
    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True)
    dev_sampler = DistributedBatchSampler(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False)
    print("samplers done!")

    train_batch_fn = Clip(
        batch_max_steps=config.batch_max_steps,
        hop_size=config.n_shift,
        aux_context_window=config.generator_params.aux_context_window)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_batch_fn,
        num_workers=config.num_workers)

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_sampler=dev_sampler,
        collate_fn=train_batch_fn,
        num_workers=config.num_workers)
    print("dataloaders done!")

    generator = PWGGenerator(**config["generator_params"])
    discriminator = PWGDiscriminator(**config["discriminator_params"])
    if world_size > 1:
        generator = DataParallel(generator)
        discriminator = DataParallel(discriminator)
    print("models done!")

    criterion_stft = MultiResolutionSTFTLoss(**config["stft_loss_params"])
    criterion_mse = nn.MSELoss()
    print("criterions done!")

    lr_schedule_g = StepDecay(**config["generator_scheduler_params"])
    gradient_clip_g = nn.ClipGradByGlobalNorm(config["generator_grad_norm"])
    optimizer_g = Adam(
        learning_rate=lr_schedule_g,
        grad_clip=gradient_clip_g,
        parameters=generator.parameters(),
        **config["generator_optimizer_params"])
    lr_schedule_d = StepDecay(**config["discriminator_scheduler_params"])
    gradient_clip_d = nn.ClipGradByGlobalNorm(config["discriminator_grad_norm"])
    optimizer_d = Adam(
        learning_rate=lr_schedule_d,
        grad_clip=gradient_clip_d,
        parameters=discriminator.parameters(),
        **config["discriminator_optimizer_params"])
    print("optimizers done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        config_name = args.config.split("/")[-1]
        # copy conf to output_dir
        shutil.copyfile(args.config, output_dir / config_name)

    updater = PWGUpdater(
        models={
            "generator": generator,
            "discriminator": discriminator,
        },
        optimizers={
            "generator": optimizer_g,
            "discriminator": optimizer_d,
        },
        criterions={
            "stft": criterion_stft,
            "mse": criterion_mse,
        },
        schedulers={
            "generator": lr_schedule_g,
            "discriminator": lr_schedule_d,
        },
        dataloader=train_dataloader,
        discriminator_train_start_steps=config.discriminator_train_start_steps,
        lambda_adv=config.lambda_adv,
        output_dir=output_dir)

    evaluator = PWGEvaluator(
        models={
            "generator": generator,
            "discriminator": discriminator,
        },
        criterions={
            "stft": criterion_stft,
            "mse": criterion_mse,
        },
        dataloader=dev_dataloader,
        lambda_adv=config.lambda_adv,
        output_dir=output_dir)
    trainer = Trainer(
        updater,
        stop_trigger=(config.train_max_steps, "iteration"),
        out=output_dir,
        profiler_options=args.profiler_options)

    if dist.get_rank() == 0:
        trainer.extend(
            evaluator, trigger=(config.eval_interval_steps, 'iteration'))
        trainer.extend(VisualDL(output_dir), trigger=(1, 'iteration'))
        trainer.extend(
            Snapshot(max_size=config.num_snapshots),
            trigger=(config.save_interval_steps, 'iteration'))

    # print(trainer.extensions.keys())
    print("Trainer Done!")
    trainer.run()


def main():
    # parse args and config and redirect to train_sp
    def str2bool(str):
        return True if str.lower() == 'true' else False

    parser = argparse.ArgumentParser(
        description="Train a ParallelWaveGAN model.")
    parser.add_argument(
        "--config", type=str, help="config file to overwrite default config.")
    parser.add_argument("--train-metadata", type=str, help="training data.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument("--verbose", type=int, default=1, help="verbose.")

    benchmark_group = parser.add_argument_group(
        'benchmark', 'arguments related to benchmark.')
    benchmark_group.add_argument(
        "--batch-size", type=int, default=8, help="batch size.")
    benchmark_group.add_argument(
        "--max-iter", type=int, default=400000, help="train max steps.")

    benchmark_group.add_argument(
        "--run-benchmark",
        type=str2bool,
        default=False,
        help="runing benchmark or not, if True, use the --batch-size and --max-iter."
    )
    benchmark_group.add_argument(
        "--profiler_options",
        type=str,
        default=None,
        help="The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\"."
    )

    args = parser.parse_args()

    with open(args.config, 'rt') as f:
        config = CfgNode(yaml.safe_load(f))

    # 增加 --batch_size --max_iter 用于 benchmark 调用
    if args.run_benchmark:
        config.batch_size = args.batch_size
        config.train_max_steps = args.max_iter

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
