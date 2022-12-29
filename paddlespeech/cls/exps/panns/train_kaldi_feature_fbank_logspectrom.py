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

import paddle
import yaml
# from paddle.audio.features import LogMelSpectrogram
from paddleaudio.utils import logger
from paddleaudio.utils import Timer

from paddlespeech.cls.models import SoundClassifier
from paddlespeech.utils.dynamic_import import dynamic_import

import kaldi_native_fbank as kaldi_feature
import numpy as np
from paddle.audio.functional import power_to_db

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--cfg_path", type=str, required=True)
args = parser.parse_args()
# yapf: enable

fbank_opts = kaldi_feature.FbankOptions()
fbank_opts.frame_opts.samp_freq = 32000
fbank_opts.frame_opts.frame_shift_ms = 10
fbank_opts.frame_opts.frame_length_ms = 32
fbank_opts.mel_opts.num_bins = 64
fbank_opts.mel_opts.low_freq = 50
fbank_opts.mel_opts.high_freq = 14000
fbank_opts.frame_opts.dither = 0
fbank_opts.use_log_fbank = False
def kaldi_feature_extractor(waveform: paddle.Tensor, **feat_conf) -> paddle.Tensor:
    kaldi_feat = []
    for b in range(waveform.shape[0]):
        extractor = kaldi_feature.OnlineFbank(fbank_opts)
        extractor.accept_waveform(fbank_opts.frame_opts.samp_freq, waveform[b].numpy().data)
        extractor.input_finished()
        num_frames = extractor.num_frames_ready
        feat = []
        for i in range(num_frames):
            feat.append(extractor.get_frame(i))
        kaldi_feat.append(feat)
    kaldi_feat = paddle.Tensor(np.array(kaldi_feat))
    kaldi_feat = paddle.transpose(kaldi_feat,[0, 2, 1])
    kaldi_feat = power_to_db(kaldi_feat, 1.0, 1e-10, None)
    kaldi_feat = paddle.transpose(kaldi_feat,[0, 2, 1])
    # print(kaldi_feat.shape)
    # print(kaldi_feat)
    return kaldi_feat

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
    training_conf = config['training']

    # Dataset
    ds_class = dynamic_import(data_conf['dataset'])
    train_ds = ds_class(**data_conf['train'])
    dev_ds = ds_class(**data_conf['dev'])
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
        use_buffer_reader=True, )

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
            waveforms, labels = batch
            feats = kaldi_feature_extractor(
                waveforms, **feat_conf
            )  # Need a padding when lengths of waveforms differ in a batch.

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
                return_list=True, )

            model.eval()
            num_corrects = 0
            num_samples = 0
            with logger.processing('Evaluation on validation dataset'):
                for batch_idx, batch in enumerate(dev_loader):
                    waveforms, labels = batch
                    feats = kaldi_feature_extractor(
                        waveforms, **feat_conf
                    )  # Need a padding when lengths of waveforms differ in a batch.

                    logits = model(feats)

                    preds = paddle.argmax(logits, axis=1)
                    num_corrects += (preds == labels).numpy().sum()
                    num_samples += feats.shape[0]

            print_msg = '[Evaluation result]'
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
