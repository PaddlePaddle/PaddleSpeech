# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.optimizer import AdamW
from paddle.optimizer.lr import OneCycleLR
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.am_batch_fn import starganv2_vc_batch_fn
from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.starganv2_vc import ASRCNN
from paddlespeech.t2s.models.starganv2_vc import Generator
from paddlespeech.t2s.models.starganv2_vc import JDCNet
from paddlespeech.t2s.models.starganv2_vc import MappingNetwork
from paddlespeech.t2s.models.starganv2_vc import StarGANv2VCEvaluator
from paddlespeech.t2s.models.starganv2_vc import StarGANv2VCUpdater
from paddlespeech.t2s.models.starganv2_vc import StyleEncoder
from paddlespeech.t2s.models.starganv2_vc.losses import adv_loss
from paddlespeech.t2s.models.starganv2_vc.losses import f0_loss
from paddlespeech.t2s.models.starganv2_vc.losses import r1_reg
from paddlespeech.t2s.training.extensions.snapshot import Snapshot
from paddlespeech.t2s.training.extensions.visualizer import VisualDL
from paddlespeech.t2s.training.seeding import seed_everything
from paddlespeech.t2s.training.trainer import Trainer
from paddlespeech.utils.env import MODEL_HOME


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

    fields = ["speech", "speech_lengths"]

    converters = {"speech": np.load}
    collate_fn = starganv2_vc_batch_fn

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

    # load model
    model_version = '1.0'
    uncompress_path = download_and_decompress(StarGANv2VC_source[model_version],
                                              MODEL_HOME)

    generator = Generator(**config['generator_params'])
    mapping_network = MappingNetwork(**config['mapping_network_params'])
    style_encoder = StyleEncoder(**config['style_encoder_params'])

    # load pretrained model
    jdc_model_dir = os.path.join(uncompress_path, 'jdcnet.pdz')
    asr_model_dir = os.path.join(uncompress_path, 'asr.pdz')

    F0_model = JDCNet(num_class=1, seq_len=192)
    F0_model.set_state_dict(paddle.load(jdc_model_dir)['main_params'])
    F0_model.eval()

    asr_model = ASRCNN(**config['asr_params'])
    asr_model.set_state_dict(paddle.load(asr_model_dir)['main_params'])
    asr_model.eval()

    if world_size > 1:
        generator = DataParallel(generator)
        discriminator = DataParallel(discriminator)
    print("models done!")

    criterion_f0 = f0_loss
    criterion_r1_reg = r1_reg
    criterion_adv = adv_loss

    print("criterions done!")

    lr_schedule_g = OneCycleLR(**config["generator_scheduler_params"])
    optimizer_g = AdamW(
        learning_rate=lr_schedule_g,
        parameters=generator.parameters(),
        **config["generator_optimizer_params"])

    lr_schedule_s = OneCycleLR(**config["style_encoder_scheduler_params"])
    optimizer_s = AdamW(
        learning_rate=lr_schedule_s,
        parameters=style_encoder.parameters(),
        **config["style_encoder_optimizer_params"])

    lr_schedule_m = OneCycleLR(**config["mapping_network_scheduler_params"])
    optimizer_m = AdamW(
        learning_rate=lr_schedule_m,
        parameters=mapping_network.parameters(),
        **config["mapping_network_optimizer_params"])

    lr_schedule_d = OneCycleLR(**config["discriminator_scheduler_params"])
    optimizer_d = AdamW(
        learning_rate=lr_schedule_d,
        parameters=discriminator.parameters(),
        **config["discriminator_optimizer_params"])
    print("optimizers done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        config_name = args.config.split("/")[-1]
        # copy conf to output_dir
        shutil.copyfile(args.config, output_dir / config_name)

    updater = StarGANv2VCUpdater(
        models={
            "generator": generator,
            "style_encoder": style_encoder,
            "mapping_network": mapping_network,
            "discriminator": discriminator,
            "F0_model": F0_model,
            "asr_model": asr_model,
        },
        optimizers={
            "generator": optimizer_g,
            "style_encoder": optimizer_s,
            "mapping_network": optimizer_m,
            "discriminator": optimizer_d,
        },
        criterions={
            "f0": criterion_f0,
            "r1_reg": criterion_r1_reg,
            "adv": criterion_adv,
        },
        schedulers={
            "generator": lr_schedule_g,
            "style_encoder": lr_schedule_s,
            "mapping_network": lr_schedule_m,
            "discriminator": lr_schedule_d,
        },
        dataloader=train_dataloader,
        g_loss_params=config.loss_params.g_loss,
        d_loss_params=config.loss_params.d_loss,
        output_dir=output_dir)

    evaluator = StarGANv2VCEvaluator(
        models={
            "generator": generator,
            "style_encoder": style_encoder,
            "mapping_network": mapping_network,
            "discriminator": discriminator,
            "F0_model": F0_model,
            "asr_model": asr_model,
        },
        criterions={
            "f0": criterion_f0,
            "r1_reg": criterion_r1_reg,
            "adv": criterion_adv,
        },
        dataloader=dev_dataloader,
        g_loss_params=config.loss_params.g_loss,
        d_loss_params=config.loss_params.d_loss,
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

    parser = argparse.ArgumentParser(description="Train a HiFiGAN model.")
    parser.add_argument("--config", type=str, help="HiFiGAN config file.")
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
