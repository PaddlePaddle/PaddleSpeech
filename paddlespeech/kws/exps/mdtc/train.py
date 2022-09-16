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

import paddle
from yacs.config import CfgNode

from paddleaudio.utils import logger
from paddleaudio.utils import Timer
from paddlespeech.kws.exps.mdtc.collate import collate_features
from paddlespeech.kws.models.loss import max_pooling_loss
from paddlespeech.kws.models.mdtc import KWSModel
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.dynamic_import import dynamic_import

if __name__ == '__main__':
    parser = default_argument_parser()
    args = parser.parse_args()

    # https://yaml.org/type/float.html
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)

    nranks = paddle.distributed.get_world_size()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    local_rank = paddle.distributed.get_rank()

    # Dataset
    ds_class = dynamic_import(config['dataset'])
    train_ds = ds_class(
        data_dir=config['data_dir'],
        mode='train',
        feat_type=config['feat_type'],
        sample_rate=config['sample_rate'],
        frame_shift=config['frame_shift'],
        frame_length=config['frame_length'],
        n_mels=config['n_mels'], )
    dev_ds = ds_class(
        data_dir=config['data_dir'],
        mode='dev',
        feat_type=config['feat_type'],
        sample_rate=config['sample_rate'],
        frame_shift=config['frame_shift'],
        frame_length=config['frame_length'],
        n_mels=config['n_mels'], )

    train_sampler = paddle.io.DistributedBatchSampler(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=False)
    train_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=config['num_workers'],
        return_list=True,
        use_buffer_reader=True,
        collate_fn=collate_features, )

    # Model
    backbone_class = dynamic_import(config['backbone'])
    backbone = backbone_class(
        stack_num=config['stack_num'],
        stack_size=config['stack_size'],
        in_channels=config['in_channels'],
        res_channels=config['res_channels'],
        kernel_size=config['kernel_size'], )
    model = KWSModel(backbone=backbone, num_keywords=config['num_keywords'])
    model = paddle.DataParallel(model)
    clip = paddle.nn.ClipGradByGlobalNorm(config['grad_clip'])
    optimizer = paddle.optimizer.Adam(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        parameters=model.parameters(),
        grad_clip=clip)
    criterion = max_pooling_loss

    steps_per_epoch = len(train_sampler)
    timer = Timer(steps_per_epoch * config['epochs'])
    timer.start()

    for epoch in range(1, config['epochs'] + 1):
        model.train()

        avg_loss = 0
        num_corrects = 0
        num_samples = 0
        for batch_idx, batch in enumerate(train_loader):
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

            if (batch_idx + 1) % config['log_freq'] == 0 and local_rank == 0:
                lr = optimizer.get_lr()
                avg_loss /= config['log_freq']
                avg_acc = num_corrects / num_samples

                print_msg = 'Epoch={}/{}, Step={}/{}'.format(
                    epoch, config['epochs'], batch_idx + 1, steps_per_epoch)
                print_msg += ' loss={:.4f}'.format(avg_loss)
                print_msg += ' acc={:.4f}'.format(avg_acc)
                print_msg += ' lr={:.6f} step/sec={:.2f} | ETA {}'.format(
                    lr, timer.timing, timer.eta)
                logger.train(print_msg)

                avg_loss = 0
                num_corrects = 0
                num_samples = 0

        if epoch % config[
                'save_freq'] == 0 and batch_idx + 1 == steps_per_epoch and local_rank == 0:
            dev_sampler = paddle.io.BatchSampler(
                dev_ds,
                batch_size=config['batch_size'],
                shuffle=False,
                drop_last=False)
            dev_loader = paddle.io.DataLoader(
                dev_ds,
                batch_sampler=dev_sampler,
                num_workers=config['num_workers'],
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
            save_dir = os.path.join(config['checkpoint_dir'],
                                    'epoch_{}'.format(epoch))
            logger.info('Saving model checkpoint to {}'.format(save_dir))
            paddle.save(model.state_dict(),
                        os.path.join(save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(save_dir, 'model.pdopt'))
