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
import yaml
from paddleaudio.utils import logger
from paddleaudio.utils import Timer

from paddlespeech.cls.models import SoundClassifier
from paddlespeech.utils.dynamic_import import dynamic_import

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--cfg_path", type=str, required=True)
args = parser.parse_args()
# yapf: enable


def _collate_features(batch):
    # (feat, label)
    # (( n_mels, length), label)
    feats = []
    labels = []
    lengths = []
    for sample in batch:
        feats.append(paddle.transpose(sample[0], perm=[1, 0]))
        lengths.append(sample[0].shape[1])
        labels.append(sample[1])

    max_length = max(lengths)
    for i in range(len(feats)):
        feats[i] = paddle.nn.functional.pad(
            feats[i], [0, max_length - feats[i].shape[0], 0, 0],
            data_format='NLC')

    return paddle.stack(feats), paddle.to_tensor(labels), paddle.to_tensor(
        lengths)


if __name__ == "__main__":
    nranks = paddle.distributed.get_world_size()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    local_rank = paddle.distributed.get_rank()

    args.cfg_path = os.path.abspath(os.path.expanduser(args.cfg_path))
    with open(args.cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    model_conf = config['model']
    data_conf = config['data']
    feat_conf = config['feature']
    feat_type = data_conf['train']['feat_type']
    training_conf = config['training']

    # Dataset

    # set audio backend, make sure paddleaudio >= 1.0.2 installed.
    paddle.audio.backends.set_backend('soundfile')

    ds_class = dynamic_import(data_conf['dataset'])
    train_ds = ds_class(**data_conf['train'], **feat_conf)
    dev_ds = ds_class(**data_conf['dev'], **feat_conf)
    train_sampler = paddle.io.DistributedBatchSampler(
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
        collate_fn=_collate_features)

    # Model
    backbone_class = dynamic_import(model_conf['backbone'])
    backbone = backbone_class(pretrained=True, extract_embedding=True)
    model = SoundClassifier(backbone, num_class=data_conf['num_classes'])
    model = paddle.DataParallel(model)
    optimizer = paddle.optimizer.Adam(
        learning_rate=training_conf['learning_rate'],
        parameters=model.parameters())
    criterion = paddle.nn.loss.CrossEntropyLoss()

    steps_per_epoch = len(train_sampler)
    timer = Timer(steps_per_epoch * training_conf['epochs'])
    timer.start()

    for epoch in range(1, training_conf['epochs'] + 1):
        model.train()

        avg_loss = 0
        num_corrects = 0
        num_samples = 0
        for batch_idx, batch in enumerate(train_loader):
            feats, labels, length = batch  # feats-->(N, length, n_mels)

            logits = model(feats)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if isinstance(optimizer._learning_rate,
                          paddle.optimizer.lr.LRScheduler):
                optimizer._learning_rate.step()
            optimizer.clear_grad()

            # Calculate loss
            avg_loss += float(loss)

            # Calculate metrics
            preds = paddle.argmax(logits, axis=1)
            num_corrects += (preds == labels).numpy().sum()
            num_samples += feats.shape[0]

            timer.count()

            if (batch_idx + 1
                ) % training_conf['log_freq'] == 0 and local_rank == 0:
                lr = optimizer.get_lr()
                avg_loss /= training_conf['log_freq']
                avg_acc = num_corrects / num_samples

                print_msg = feat_type + ' Epoch={}/{}, Step={}/{}'.format(
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

        if epoch % training_conf[
                'save_freq'] == 0 and batch_idx + 1 == steps_per_epoch and local_rank == 0:
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
                collate_fn=_collate_features)

            model.eval()
            num_corrects = 0
            num_samples = 0
            with logger.processing('Evaluation on validation dataset'):
                for batch_idx, batch in enumerate(dev_loader):
                    feats, labels, length = batch
                    logits = model(feats)

                    preds = paddle.argmax(logits, axis=1)
                    num_corrects += (preds == labels).numpy().sum()
                    num_samples += feats.shape[0]

            print_msg = '[Evaluation result] ' + str(feat_type)
            print_msg += ' dev_acc={:.4f}'.format(num_corrects / num_samples)

            logger.eval(print_msg)

            # Save model
            save_dir = os.path.join(training_conf['checkpoint_dir'],
                                    'epoch_{}'.format(epoch))
            logger.info('Saving model checkpoint to {}'.format(save_dir))
            paddle.save(model.state_dict(),
                        os.path.join(save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(save_dir, 'model.pdopt'))
