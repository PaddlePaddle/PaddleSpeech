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
import time

import numpy as np
import paddle
from paddle.io import BatchSampler
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from yacs.config import CfgNode

from paddleaudio.compliance.librosa import melspectrogram
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.io.augment import build_augment_pipeline
from paddlespeech.vector.io.augment import waveform_augment
from paddlespeech.vector.io.batch import batch_pad_right
from paddlespeech.vector.io.batch import feature_normalize
from paddlespeech.vector.io.batch import waveform_collate_fn
from paddlespeech.vector.io.dataset import CSVDataset
from paddlespeech.vector.models.ecapa_tdnn import EcapaTdnn
from paddlespeech.vector.modules.loss import AdditiveAngularMargin
from paddlespeech.vector.modules.loss import LogSoftmaxWrapper
from paddlespeech.vector.modules.sid_model import SpeakerIdetification
from paddlespeech.vector.training.scheduler import CyclicLRScheduler
from paddlespeech.vector.training.seeding import seed_everything
from paddlespeech.vector.utils.time import Timer

logger = Log(__name__).getlog()


def main(args, config):
    # stage0: set the training device, cpu or gpu
    paddle.set_device(args.device)

    # stage1: we must call the paddle.distributed.init_parallel_env() api at the begining
    paddle.distributed.init_parallel_env()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    # set the random seed, it is a must for multiprocess training
    seed_everything(config.seed)

    # stage2: data prepare, such vox1 and vox2 data, and augment noise data and pipline
    # note: some cmd must do in rank==0, so wo will refactor the data prepare code
    train_dataset = CSVDataset(
        csv_path=os.path.join(args.data_dir, "vox/csv/train.csv"),
        label2id_path=os.path.join(args.data_dir, "vox/meta/label2id.txt"))
    dev_dataset = CSVDataset(
        csv_path=os.path.join(args.data_dir, "vox/csv/dev.csv"),
        label2id_path=os.path.join(args.data_dir, "vox/meta/label2id.txt"))

    if config.augment:
        augment_pipeline = build_augment_pipeline(target_dir=args.data_dir)
    else:
        augment_pipeline = []

    # stage3: build the dnn backbone model network
    ecapa_tdnn = EcapaTdnn(**config.model)

    # stage4: build the speaker verification train instance with backbone model
    model = SpeakerIdetification(
        backbone=ecapa_tdnn, num_class=config.num_speakers)

    # stage5: build the optimizer, we now only construct the AdamW optimizer
    #         140000 is single gpu steps
    #         so, in multi-gpu mode, wo reduce the step_size to 140000//nranks to enable CyclicLRScheduler
    lr_schedule = CyclicLRScheduler(
        base_lr=config.learning_rate, max_lr=1e-3, step_size=140000 // nranks)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_schedule, parameters=model.parameters())

    # stage6: build the loss function, we now only support LogSoftmaxWrapper
    criterion = LogSoftmaxWrapper(
        loss_fn=AdditiveAngularMargin(margin=0.2, scale=30))

    # stage7: confirm training start epoch
    #         if pre-trained model exists, start epoch confirmed by the pre-trained model
    start_epoch = 0
    if args.load_checkpoint:
        logger.info("load the check point")
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
                logger.info(f'Checkpoint loaded from {args.load_checkpoint}')
        except FileExistsError:
            if local_rank == 0:
                logger.info('Train from scratch.')

        try:
            start_epoch = int(args.load_checkpoint[-1])
            logger.info(f'Restore training from epoch {start_epoch}.')
        except ValueError:
            pass

    # stage8: we build the batch sampler for paddle.DataLoader
    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=waveform_collate_fn,
        return_list=True,
        use_buffer_reader=True, )

    # stage9: start to train
    #         we will comment the training process
    steps_per_epoch = len(train_sampler)
    timer = Timer(steps_per_epoch * config.epochs)
    last_saved_epoch = ""
    timer.start()

    for epoch in range(start_epoch + 1, config.epochs + 1):
        # at the begining, model must set to train mode
        model.train()

        avg_loss = 0
        num_corrects = 0
        num_samples = 0
        train_reader_cost = 0.0
        train_feat_cost = 0.0
        train_run_cost = 0.0

        reader_start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            train_reader_cost += time.time() - reader_start

            # stage 9-1: batch data is audio sample points and speaker id label
            feat_start = time.time()
            waveforms, labels = batch['waveforms'], batch['labels']
            waveforms, lengths = batch_pad_right(waveforms.numpy())
            waveforms = paddle.to_tensor(waveforms)

            # stage 9-2: audio sample augment method, which is done on the audio sample point
            #            the original wavefrom and the augmented waveform is concatented in a batch
            #            eg. five augment method in the augment pipeline
            #                the final data nums is batch_size * [five + one] 
            #                -> five augmented waveform batch plus one original batch waveform
            if len(augment_pipeline) != 0:
                waveforms = waveform_augment(waveforms, augment_pipeline)
                labels = paddle.concat(
                    [labels for i in range(len(augment_pipeline) + 1)])

            # stage 9-3: extract the audio feats,such fbank, mfcc, spectrogram
            feats = []
            for waveform in waveforms.numpy():
                feat = melspectrogram(
                    x=waveform,
                    sr=config.sr,
                    n_mels=config.n_mels,
                    window_size=config.window_size,
                    hop_length=config.hop_size)
                feats.append(feat)
            feats = paddle.to_tensor(np.asarray(feats))

            # stage 9-4: feature normalize, which help converge and imporve the performance
            feats = feature_normalize(
                feats, mean_norm=True, std_norm=False)  # Features normalization
            train_feat_cost += time.time() - feat_start

            # stage 9-5: model forward, such ecapa-tdnn, x-vector
            train_start = time.time()
            logits = model(feats)

            # stage 9-6: loss function criterion, such AngularMargin, AdditiveAngularMargin
            loss = criterion(logits, labels)

            # stage 9-7: update the gradient and clear the gradient cache
            loss.backward()
            optimizer.step()
            if isinstance(optimizer._learning_rate,
                          paddle.optimizer.lr.LRScheduler):
                optimizer._learning_rate.step()
            optimizer.clear_grad()

            # stage 9-8: Calculate average loss per batch
            avg_loss = loss.item()

            # stage 9-9: Calculate metrics, which is one-best accuracy
            preds = paddle.argmax(logits, axis=1)
            num_corrects += (preds == labels).numpy().sum()
            num_samples += feats.shape[0]
            train_run_cost += time.time() - train_start
            timer.count()  # step plus one in timer

            # stage 9-10: print the log information only on 0-rank per log-freq batchs
            if (batch_idx + 1) % config.log_interval == 0 and local_rank == 0:
                lr = optimizer.get_lr()
                avg_loss /= config.log_interval
                avg_acc = num_corrects / num_samples

                print_msg = 'Train Epoch={}/{}, Step={}/{}'.format(
                    epoch, config.epochs, batch_idx + 1, steps_per_epoch)
                print_msg += ' loss={:.4f}'.format(avg_loss)
                print_msg += ' acc={:.4f}'.format(avg_acc)
                print_msg += ' avg_reader_cost: {:.5f} sec,'.format(
                    train_reader_cost / config.log_interval)
                print_msg += ' avg_feat_cost: {:.5f} sec,'.format(
                    train_feat_cost / config.log_interval)
                print_msg += ' avg_train_cost: {:.5f} sec,'.format(
                    train_run_cost / config.log_interval)

                print_msg += ' lr={:.4E} step/sec={:.2f} ips:{:.5f}| ETA {}'.format(
                    lr, timer.timing, timer.ips, timer.eta)
                logger.info(print_msg)

                avg_loss = 0
                num_corrects = 0
                num_samples = 0
                train_reader_cost = 0.0
                train_feat_cost = 0.0
                train_run_cost = 0.0

            reader_start = time.time()

        # stage 9-11: save the model parameters only on 0-rank per save-freq batchs
        if epoch % config.save_interval == 0 and batch_idx + 1 == steps_per_epoch:
            if local_rank != 0:
                paddle.distributed.barrier(
                )  # Wait for valid step in main process
                continue  # Resume trainning on other process

            # stage 9-12: construct the valid dataset dataloader
            dev_sampler = BatchSampler(
                dev_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False)
            dev_loader = DataLoader(
                dev_dataset,
                batch_sampler=dev_sampler,
                collate_fn=waveform_collate_fn,
                num_workers=config.num_workers,
                return_list=True, )

            # set the model to eval mode
            model.eval()
            num_corrects = 0
            num_samples = 0

            # stage 9-13: evaluation the valid dataset batch data
            logger.info('Evaluate on validation dataset')
            with paddle.no_grad():
                for batch_idx, batch in enumerate(dev_loader):
                    waveforms, labels = batch['waveforms'], batch['labels']

                    feats = []
                    for waveform in waveforms.numpy():
                        feat = melspectrogram(
                            x=waveform,
                            sr=config.sr,
                            n_mels=config.n_mels,
                            window_size=config.window_size,
                            hop_length=config.hop_size)
                        feats.append(feat)

                    feats = paddle.to_tensor(np.asarray(feats))
                    feats = feature_normalize(
                        feats, mean_norm=True, std_norm=False)
                    logits = model(feats)

                    preds = paddle.argmax(logits, axis=1)
                    num_corrects += (preds == labels).numpy().sum()
                    num_samples += feats.shape[0]

            print_msg = '[Evaluation result]'
            print_msg += ' dev_acc={:.4f}'.format(num_corrects / num_samples)
            logger.info(print_msg)

            # stage 9-14: Save model parameters
            save_dir = os.path.join(args.checkpoint_dir,
                                    'epoch_{}'.format(epoch))
            last_saved_epoch = os.path.join('epoch_{}'.format(epoch),
                                            "model.pdparams")
            logger.info('Saving model checkpoint to {}'.format(save_dir))
            paddle.save(model.state_dict(),
                        os.path.join(save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(save_dir, 'model.pdopt'))

            if nranks > 1:
                paddle.distributed.barrier()  # Main process

    # stage 10: create the final trained model.pdparams with soft link
    if local_rank == 0:
        final_model = os.path.join(args.checkpoint_dir, "model.pdparams")
        logger.info(f"we will create the final model: {final_model}")
        if os.path.islink(final_model):
            logger.info(
                f"An {final_model} already exists, we will rm is and create it again"
            )
            os.unlink(final_model)
        os.symlink(last_saved_epoch, final_model)


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device',
                        choices=['cpu', 'gpu'],
                        default="cpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--config",
                        default=None,
                        type=str,
                        help="configuration file")
    parser.add_argument("--data-dir",
                        default="./data/",
                        type=str,
                        help="data directory")
    parser.add_argument("--load-checkpoint",
                        type=str,
                        default=None,
                        help="Directory to load model checkpoint to contiune trainning.")
    parser.add_argument("--checkpoint-dir",
                        type=str,
                        default='./checkpoint',
                        help="Directory to save model checkpoints.")

    args = parser.parse_args()
    # yapf: enable

    # https://yaml.org/type/float.html
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)

    config.freeze()
    print(config)

    main(args, config)
