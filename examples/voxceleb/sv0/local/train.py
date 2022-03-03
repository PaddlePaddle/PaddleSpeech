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
import os

import paddle
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

from paddleaudio.datasets.voxceleb import VoxCeleb1
from paddlespeech.vector.datasets.batch import waveform_collate_fn
from paddlespeech.vector.layers.loss import AdditiveAngularMargin
from paddlespeech.vector.layers.loss import LogSoftmaxWrapper
from paddlespeech.vector.layers.lr import CyclicLRScheduler
from paddlespeech.vector.models.ecapa_tdnn import EcapaTdnn
from paddlespeech.vector.training.sid_model import SpeakerIdetification


def main(args):
    # stage0: set the training device, cpu or gpu
    paddle.set_device(args.device)

    # stage1: we must call the paddle.distributed.init_parallel_env() api at the begining
    paddle.distributed.init_parallel_env()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()

    # stage2: data prepare
    # note: some cmd must do in rank==0
    train_ds = VoxCeleb1('train', target_dir=args.data_dir)
    dev_ds = VoxCeleb1('dev', target_dir=args.data_dir)

    # stage3: build the dnn backbone model network
    model_conf = {
        "input_size": 80,
        "channels": [1024, 1024, 1024, 1024, 3072],
        "kernel_sizes": [5, 3, 3, 3, 1],
        "dilations": [1, 2, 3, 4, 1],
        "attention_channels": 128,
        "lin_neurons": 192,
    }
    ecapa_tdnn = EcapaTdnn(**model_conf)

    # stage4: build the speaker verification train instance with backbone model
    model = SpeakerIdetification(
        backbone=ecapa_tdnn, num_class=VoxCeleb1.num_speakers)

    # stage5: build the optimizer, we now only construct the AdamW optimizer
    lr_schedule = CyclicLRScheduler(
        base_lr=args.learning_rate, max_lr=1e-3, step_size=140000 // nranks)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_schedule, parameters=model.parameters())

    # stage6: build the loss function, we now only support LogSoftmaxWrapper
    criterion = LogSoftmaxWrapper(
        loss_fn=AdditiveAngularMargin(margin=0.2, scale=30))

    # stage7: confirm training start epoch
    #         if pre-trained model exists, start epoch confirmed by the pre-trained model
    start_epoch = 0
    if args.load_checkpoint:
        args.load_checkpoint = os.path.abspath(
            os.path.expanduser(args.load_checkpoint))
        try:
            # load model checkpoint
            state_dict = paddle.load(
                os.path.join(args.load_checkpoint, 'model.pdparams'))
            model.set_state_dict(state_dict)

            # load optimizer checkpoint
            state_dict = paddle.load(
                os.path.join(args.load_checkpoint, 'model.pdopt'))
            optimizer.set_state_dict(state_dict)
            if local_rank == 0:
                print(f'Checkpoint loaded from {args.load_checkpoint}')
        except FileExistsError:
            if local_rank == 0:
                print('Train from scratch.')

        try:
            start_epoch = int(args.load_checkpoint[-1])
            print(f'Restore training from epoch {start_epoch}.')
        except ValueError:
            pass

    # stage8: we build the batch sampler for paddle.DataLoader
    train_sampler = DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=waveform_collate_fn,
        return_list=True,
        use_buffer_reader=True, )


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device',
                        choices=['cpu', 'gpu'],
                        default="cpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--data-dir",
                        default="./data/",
                        type=str,
                        help="data directory")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-8,
                        help="Learning rate used to train with warmup.")
    parser.add_argument("--load_checkpoint",
                        type=str,
                        default=None,
                        help="Directory to load model checkpoint to contiune trainning.")
    parser.add_argument("--batch_size",
                        type=int, default=64,
                        help="Total examples' number in batch for training.")
    parser.add_argument("--num_workers",
                        type=int,
                        default=0,
                        help="Number of workers in dataloader.")

    args = parser.parse_args()
    # yapf: enable

    main(args)
