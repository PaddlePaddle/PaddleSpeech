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
import paddle.nn.functional as F
import yaml
from paddle.audio.features import LogMelSpectrogram
from paddleaudio.backends import soundfile_load as load_audio
from paddleaudio.utils import logger

from paddlespeech.cls.models import SoundClassifier
from paddlespeech.utils.dynamic_import import dynamic_import
#from paddleaudio.features import LogMelSpectrogram

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

def extract_features(file: str, **feat_conf) -> paddle.Tensor:
    file = os.path.abspath(os.path.expanduser(file))
    waveform, _ = load_audio(file, sr=feat_conf['sr'], normal=False)
    print(waveform.size)
    print(waveform)

    extractor = LogMelSpectrogram(**feat_conf)
    feat = extractor(paddle.to_tensor(waveform).unsqueeze(0))
    feat = paddle.transpose(feat, [0, 2, 1])
    print(feat.shape)
    print(feat)
    return feat

def extract_kaldi_features(file: str, **feat_conf) -> paddle.Tensor:
    file = os.path.abspath(os.path.expanduser(file))
    waveform, _ = load_audio(file, sr=feat_conf['sr'], normal=False)
    print(waveform.size)
    print(waveform)

    extractor = kaldi_feature.OnlineFbank(fbank_opts)
    extractor.accept_waveform(fbank_opts.frame_opts.samp_freq, waveform.data)
    extractor.input_finished()
    num_frames = extractor.num_frames_ready
    kaldi_feat = []
    feat = []
    for i in range(num_frames):
        feat.append(extractor.get_frame(i))
    kaldi_feat.append(feat)
    kaldi_feat = paddle.Tensor(np.array(kaldi_feat))
    print(kaldi_feat)
    kaldi_feat = paddle.transpose(kaldi_feat,[0, 2, 1])
    kaldi_feat = power_to_db(kaldi_feat, 1.0, 1e-10, None)
    kaldi_feat = paddle.transpose(kaldi_feat,[0, 2, 1])
    print(kaldi_feat.shape)
    print(kaldi_feat)
    return kaldi_feat

def feature_extractor(waveform: paddle.Tensor, **feat_conf) -> paddle.Tensor:
    extractor = LogMelSpectrogram(**feat_conf)
    feat = extractor(waveform)
    feat = paddle.transpose(feat, [0, 2, 1])
    # print(feat.shape)
    # print(feat)
    return feat

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

if __name__ == '__main__':

    args.cfg_path = os.path.abspath(os.path.expanduser(args.cfg_path))
    with open(args.cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    model_conf = config['model']
    data_conf = config['data']
    feat_conf = config['feature']
    predicting_conf = config['predicting']

    ds_class = dynamic_import(data_conf['dataset'])
    backbone_class = dynamic_import(model_conf['backbone'])

    model = SoundClassifier(
        backbone=backbone_class(pretrained=False, extract_embedding=True),
        num_class=len(ds_class.label_list))
    model.set_state_dict(paddle.load(predicting_conf['checkpoint']))
    model.eval()

    feat = extract_features(predicting_conf['audio_file'], **feat_conf)
    #feat = extract_kaldi_features(predicting_conf['audio_file'], **feat_conf)
    logits = model(feat)
    probs = F.softmax(logits, axis=1).numpy()

    sorted_indices = (-probs[0]).argsort()

    msg = f"[{predicting_conf['audio_file']}]\n"
    for idx in sorted_indices[:predicting_conf['top_k']]:
        msg += f'{ds_class.label_list[idx]}: {probs[0][idx]}\n'
    logger.info(msg)

    #exit(1)

    training_conf = config['training']
    dev_ds = ds_class(**data_conf['dev'])
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
    num_corrects = 0
    num_samples = 0
    with logger.processing('Evaluation on validation dataset'):
        for batch_idx, batch in enumerate(dev_loader):
            waveforms, labels = batch
            feats = feature_extractor(waveforms, **feat_conf)
            #feats = kaldi_feature_extractor(waveforms, **feat_conf)

            logits = model(feats)

            preds = paddle.argmax(logits, axis=1)
            num_corrects += (preds == labels).numpy().sum()
            num_samples += feats.shape[0]

    print_msg = '[Evaluation result]'
    print_msg += ' dev_acc={:.4f}'.format(num_corrects / num_samples)

    logger.eval(print_msg)
