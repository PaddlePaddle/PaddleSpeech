# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.optimizer import AdamW
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.am_batch_fn import vits_multi_spk_batch_fn
from paddlespeech.t2s.datasets.am_batch_fn import vits_single_spk_batch_fn
from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.datasets.sampler import ErnieSATSampler
from paddlespeech.t2s.models.vits import VITS
from paddlespeech.t2s.models.vits import VITSEvaluator
from paddlespeech.t2s.models.vits import VITSUpdater
from paddlespeech.t2s.modules.losses import DiscriminatorAdversarialLoss
from paddlespeech.t2s.modules.losses import FeatureMatchLoss
from paddlespeech.t2s.modules.losses import GeneratorAdversarialLoss
from paddlespeech.t2s.modules.losses import KLDivergenceLoss
from paddlespeech.t2s.modules.losses import MelSpectrogramLoss
from paddlespeech.t2s.training.extensions.snapshot import Snapshot
from paddlespeech.t2s.training.extensions.visualizer import VisualDL
from paddlespeech.t2s.training.optimizer import scheduler_classes
from paddlespeech.t2s.training.seeding import seed_everything
from paddlespeech.t2s.training.trainer import Trainer
from paddlespeech.t2s.utils import str2bool


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

    fields = ["text", "text_lengths", "feats", "feats_lengths", "wave"]

    converters = {
        "wave": np.load,
        "feats": np.load,
    }
    spk_num = None
    if args.speaker_dict is not None:
        print("multiple speaker vits!")
        collate_fn = vits_multi_spk_batch_fn
        with open(args.speaker_dict, 'rt', encoding='utf-8') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
        fields += ["spk_id"]
    elif args.voice_cloning:
        print("Training voice cloning!")
        collate_fn = vits_multi_spk_batch_fn
        fields += ["spk_emb"]
        converters["spk_emb"] = np.load
    else:
        print("single speaker vits!")
        collate_fn = vits_single_spk_batch_fn
    print("spk_num:", spk_num)

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
    train_sampler = ErnieSATSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True)
    dev_sampler = ErnieSATSampler(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False)
    print("samplers done!")

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers)

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_sampler=dev_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers)
    print("dataloaders done!")

    with open(args.phones_dict, 'rt', encoding='utf-8') as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    odim = config.n_fft // 2 + 1
    config["model"]["generator_params"]["spks"] = spk_num
    model = VITS(idim=vocab_size, odim=odim, **config["model"])
    gen_parameters = model.generator.parameters()
    dis_parameters = model.discriminator.parameters()
    if world_size > 1:
        model = DataParallel(model)
        gen_parameters = model._layers.generator.parameters()
        dis_parameters = model._layers.discriminator.parameters()

    print("model done!")

    # loss
    criterion_mel = MelSpectrogramLoss(
        **config["mel_loss_params"], )
    criterion_feat_match = FeatureMatchLoss(
        **config["feat_match_loss_params"], )
    criterion_gen_adv = GeneratorAdversarialLoss(
        **config["generator_adv_loss_params"], )
    criterion_dis_adv = DiscriminatorAdversarialLoss(
        **config["discriminator_adv_loss_params"], )
    criterion_kl = KLDivergenceLoss()

    print("criterions done!")

    lr_schedule_g = scheduler_classes[config["generator_scheduler"]](
        **config["generator_scheduler_params"])
    optimizer_g = AdamW(
        learning_rate=lr_schedule_g,
        parameters=gen_parameters,
        **config["generator_optimizer_params"])

    lr_schedule_d = scheduler_classes[config["discriminator_scheduler"]](
        **config["discriminator_scheduler_params"])
    optimizer_d = AdamW(
        learning_rate=lr_schedule_d,
        parameters=dis_parameters,
        **config["discriminator_optimizer_params"])

    print("optimizers done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        config_name = args.config.split("/")[-1]
        # copy conf to output_dir
        shutil.copyfile(args.config, output_dir / config_name)

    updater = VITSUpdater(
        model=model,
        optimizers={
            "generator": optimizer_g,
            "discriminator": optimizer_d,
        },
        criterions={
            "mel": criterion_mel,
            "feat_match": criterion_feat_match,
            "gen_adv": criterion_gen_adv,
            "dis_adv": criterion_dis_adv,
            "kl": criterion_kl,
        },
        schedulers={
            "generator": lr_schedule_g,
            "discriminator": lr_schedule_d,
        },
        dataloader=train_dataloader,
        lambda_adv=config.lambda_adv,
        lambda_mel=config.lambda_mel,
        lambda_kl=config.lambda_kl,
        lambda_feat_match=config.lambda_feat_match,
        lambda_dur=config.lambda_dur,
        generator_first=config.generator_first,
        output_dir=output_dir)

    evaluator = VITSEvaluator(
        model=model,
        criterions={
            "mel": criterion_mel,
            "feat_match": criterion_feat_match,
            "gen_adv": criterion_gen_adv,
            "dis_adv": criterion_dis_adv,
            "kl": criterion_kl,
        },
        dataloader=dev_dataloader,
        lambda_adv=config.lambda_adv,
        lambda_mel=config.lambda_mel,
        lambda_kl=config.lambda_kl,
        lambda_feat_match=config.lambda_feat_match,
        lambda_dur=config.lambda_dur,
        generator_first=config.generator_first,
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

    parser = argparse.ArgumentParser(description="Train a VITS model.")
    parser.add_argument("--config", type=str, help="VITS config file")
    parser.add_argument("--train-metadata", type=str, help="training data.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument(
        "--phones-dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--speaker-dict",
        type=str,
        default=None,
        help="speaker id map file for multiple speaker model.")

    parser.add_argument(
        "--voice-cloning",
        type=str2bool,
        default=False,
        help="whether training voice cloning model.")

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
