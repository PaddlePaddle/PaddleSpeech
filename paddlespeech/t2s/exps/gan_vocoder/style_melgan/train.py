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
from paddle.optimizer.lr import MultiStepDecay
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.datasets.vocoder_batch_fn import Clip
from paddlespeech.t2s.models.melgan import StyleMelGANDiscriminator
from paddlespeech.t2s.models.melgan import StyleMelGANEvaluator
from paddlespeech.t2s.models.melgan import StyleMelGANGenerator
from paddlespeech.t2s.models.melgan import StyleMelGANUpdater
from paddlespeech.t2s.modules.losses import DiscriminatorAdversarialLoss
from paddlespeech.t2s.modules.losses import GeneratorAdversarialLoss
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

    if "aux_context_window" in config.generator_params:
        aux_context_window = config.generator_params.aux_context_window
    else:
        aux_context_window = 0
    train_batch_fn = Clip(
        batch_max_steps=config.batch_max_steps,
        hop_size=config.n_shift,
        aux_context_window=aux_context_window)

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

    generator = StyleMelGANGenerator(**config["generator_params"])
    discriminator = StyleMelGANDiscriminator(**config["discriminator_params"])
    if world_size > 1:
        generator = DataParallel(generator)
        discriminator = DataParallel(discriminator)
    print("models done!")
    criterion_stft = MultiResolutionSTFTLoss(**config["stft_loss_params"])

    criterion_gen_adv = GeneratorAdversarialLoss(
        **config["generator_adv_loss_params"])
    criterion_dis_adv = DiscriminatorAdversarialLoss(
        **config["discriminator_adv_loss_params"])
    print("criterions done!")

    lr_schedule_g = MultiStepDecay(**config["generator_scheduler_params"])
    # Compared to multi_band_melgan.v1 config, Adam optimizer without gradient norm is used
    generator_grad_norm = config["generator_grad_norm"]
    gradient_clip_g = nn.ClipGradByGlobalNorm(
        generator_grad_norm) if generator_grad_norm > 0 else None
    print("gradient_clip_g:", gradient_clip_g)

    optimizer_g = Adam(
        learning_rate=lr_schedule_g,
        grad_clip=gradient_clip_g,
        parameters=generator.parameters(),
        **config["generator_optimizer_params"])
    lr_schedule_d = MultiStepDecay(**config["discriminator_scheduler_params"])
    discriminator_grad_norm = config["discriminator_grad_norm"]
    gradient_clip_d = nn.ClipGradByGlobalNorm(
        discriminator_grad_norm) if discriminator_grad_norm > 0 else None
    print("gradient_clip_d:", gradient_clip_d)
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

    updater = StyleMelGANUpdater(
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
            "gen_adv": criterion_gen_adv,
            "dis_adv": criterion_dis_adv,
        },
        schedulers={
            "generator": lr_schedule_g,
            "discriminator": lr_schedule_d,
        },
        dataloader=train_dataloader,
        discriminator_train_start_steps=config.discriminator_train_start_steps,
        lambda_adv=config.lambda_adv,
        output_dir=output_dir)

    evaluator = StyleMelGANEvaluator(
        models={
            "generator": generator,
            "discriminator": discriminator,
        },
        criterions={
            "stft": criterion_stft,
            "gen_adv": criterion_gen_adv,
            "dis_adv": criterion_dis_adv,
        },
        dataloader=dev_dataloader,
        lambda_adv=config.lambda_adv,
        output_dir=output_dir)

    trainer = Trainer(
        updater,
        stop_trigger=(config.train_max_steps, "iteration"),
        out=output_dir)

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

    parser = argparse.ArgumentParser(description="Train a Style MelGAN model.")
    parser.add_argument("--config", type=str, help="Style MelGAN config file.")
    parser.add_argument("--train-metadata", type=str, help="training data.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")

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
