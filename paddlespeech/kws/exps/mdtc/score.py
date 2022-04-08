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
import sys
import time

import paddle
from mdtc import KWSModel
from mdtc import MDTC
from tqdm import tqdm

from paddleaudio.datasets import HeySnips


def collate_features(batch):
    # (key, feat, label) in one sample
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
    test_ds = HeySnips(
        mode='test', feat_type='kaldi_fbank', sample_rate=16000, **feat_conf)
    test_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=32, drop_last=False)
    test_loader = paddle.io.DataLoader(
        test_ds,
        batch_sampler=test_sampler,
        num_workers=16,
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
    # kws_checkpoint = '/ssd3/chenxiaojie06/PaddleSpeech/DeepSpeech/paddlespeech/kws/models/checkpoint/epoch_10_0.8903940343290826/model.pdparams'
    kws_checkpoint = os.path.join(
        os.path.abspath(sys.argv[1]), 'model.pdparams')
    model.set_state_dict(paddle.load(kws_checkpoint))
    model.eval()

    score_abs_path = os.path.join(os.path.abspath(sys.argv[1]), 'score.txt')
    with paddle.no_grad(), open(score_abs_path, 'w', encoding='utf8') as fout:
        for batch_idx, batch in enumerate(
                tqdm(test_loader, total=len(test_loader))):
            keys, feats, labels, lengths = batch
            logits = model(feats)
            num_keywords = logits.shape[2]
            for i in range(len(keys)):
                key = keys[i]
                score = logits[i][:lengths[i]]
                for keyword_i in range(num_keywords):
                    keyword_scores = score[:, keyword_i]
                    score_frames = ' '.join(
                        ['{:.6f}'.format(x) for x in keyword_scores.tolist()])
                    fout.write(
                        '{} {} {}\n'.format(key, keyword_i, score_frames))

    print('Scores saved to: {}'.format(score_abs_path))
