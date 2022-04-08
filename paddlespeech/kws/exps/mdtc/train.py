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
import os
import time

import paddle
from loss import max_pooling_loss
from mdtc import KWSModel
from mdtc import MDTC

from paddleaudio.datasets import HeySnips
from paddleaudio.utils import logger
from paddleaudio.utils import Timer


def collate_features(batch):
    # (key, feat, label)
    collate_start = time.time()
    keys = []
    feats = []
    labels = []
    lengths = []
    for sample in batch:
        keys.append(sample[0])
        feats.append(sample[1])
        labels.append(sample[2])
        lengths.append(sample[1].shape[0])

    max_length = max(lengths)
    for i in range(len(feats)):
        feats[i] = paddle.nn.functional.pad(
            feats[i], [0, max_length - feats[i].shape[0], 0, 0],
            data_format='NLC')

    return keys, paddle.stack(feats), paddle.to_tensor(
        labels), paddle.to_tensor(lengths)


if __name__ == '__main__':
    # Dataset
    feat_conf = {
        # 'n_mfcc': 80,
        'n_mels': 80,
        'frame_shift': 10,
        'frame_length': 25,
        # 'dither': 1.0,
    }
    data_dir = '/ssd1/chenxiaojie06/datasets/hey_snips/hey_snips_research_6k_en_train_eval_clean_ter'
    train_ds = HeySnips(
        data_dir=data_dir,
        mode='train',
        feat_type='kaldi_fbank',
        sample_rate=16000,
        **feat_conf)
    dev_ds = HeySnips(
        data_dir=data_dir,
        mode='dev',
        feat_type='kaldi_fbank',
        sample_rate=16000,
        **feat_conf)

    training_conf = {
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 0.00005,
        'num_workers': 16,
        'batch_size': 100,
        'checkpoint_dir': './checkpoint',
        'save_freq': 10,
        'log_freq': 10,
    }

    train_sampler = paddle.io.BatchSampler(
        train_ds,
        batch_size=training_conf['batch_size'],
        shuffle=True,
        drop_last=False)
    train_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=training_conf['num_workers'],
        return_list=True,
        use_buffer_reader=True,
        collate_fn=collate_features, )

    # Model
    backbone = MDTC(
        stack_num=3,
        stack_size=4,
        in_channels=80,
        res_channels=32,
        kernel_size=5,
        causal=True, )
    model = KWSModel(backbone=backbone, num_keywords=1)
    model = paddle.DataParallel(model)
    clip = paddle.nn.ClipGradByGlobalNorm(5.0)
    optimizer = paddle.optimizer.Adam(
        learning_rate=training_conf['learning_rate'],
        weight_decay=training_conf['weight_decay'],
        parameters=model.parameters(),
        grad_clip=clip)
    criterion = max_pooling_loss

    steps_per_epoch = len(train_sampler)
    timer = Timer(steps_per_epoch * training_conf['epochs'])
    timer.start()

    for epoch in range(1, training_conf['epochs'] + 1):
        model.train()

        avg_loss = 0
        num_corrects = 0
        num_samples = 0
        batch_start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            # print('Fetch one batch: {:.4f}'.format(time.time()-batch_start))
            keys, feats, labels, lengths = batch
            logits = model(feats)
            loss, corrects, acc = criterion(logits, labels, lengths)
            loss.backward()
            optimizer.step()
            if isinstance(optimizer._learning_rate,
                          paddle.optimizer.lr.LRScheduler):
                optimizer._learning_rate.step()
            optimizer.clear_grad()

            # Calculate loss
            avg_loss += loss.numpy()[0]

            # Calculate metrics
            num_corrects += corrects
            num_samples += feats.shape[0]

            timer.count()

            if (batch_idx + 1) % training_conf['log_freq'] == 0:
                lr = optimizer.get_lr()
                avg_loss /= training_conf['log_freq']
                avg_acc = num_corrects / num_samples

                print_msg = 'Epoch={}/{}, Step={}/{}'.format(
                    epoch, training_conf['epochs'], batch_idx + 1,
                    steps_per_epoch)
                print_msg += ' loss={:.4f}'.format(avg_loss)
                print_msg += ' acc={:.4f}'.format(avg_acc)
                print_msg += ' lr={:.6f} step/sec={:.2f} | ETA {}'.format(
                    lr, timer.timing, timer.eta)
                logger.train(print_msg)

                avg_loss = 0
                num_corrects = 0
                num_samples = 0
            batch_start = time.time()

        if epoch % training_conf[
                'save_freq'] == 0 and batch_idx + 1 == steps_per_epoch:
            dev_sampler = paddle.io.BatchSampler(
                dev_ds,
                batch_size=training_conf['batch_size'],
                shuffle=False,
                drop_last=False)
            dev_loader = paddle.io.DataLoader(
                dev_ds,
                batch_sampler=dev_sampler,
                num_workers=training_conf['num_workers'],
                return_list=True,
                use_buffer_reader=True,
                collate_fn=collate_features, )

            model.eval()
            num_corrects = 0
            num_samples = 0
            with logger.processing('Evaluation on validation dataset'):
                for batch_idx, batch in enumerate(dev_loader):
                    keys, feats, labels, lengths = batch
                    logits = model(feats)
                    loss, corrects, acc = criterion(logits, labels, lengths)
                    num_corrects += corrects
                    num_samples += feats.shape[0]

            eval_acc = num_corrects / num_samples
            print_msg = '[Evaluation result]'
            print_msg += ' dev_acc={:.4f}'.format(eval_acc)

            logger.eval(print_msg)

            # Save model
            save_dir = os.path.join(training_conf['checkpoint_dir'],
                                    'epoch_{}_{:.4f}'.format(epoch, eval_acc))
            logger.info('Saving model checkpoint to {}'.format(save_dir))
            paddle.save(model.state_dict(),
                        os.path.join(save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(save_dir, 'model.pdopt'))
