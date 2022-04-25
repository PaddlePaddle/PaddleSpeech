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
# Modified from wekws(https://github.com/wenet-e2e/wekws)
import argparse
import os

import paddle
import yaml
from tqdm import tqdm

from paddlespeech.kws.exps.mdtc.collate import collate_features
from paddlespeech.kws.models.mdtc import KWSModel
from paddlespeech.s2t.utils.dynamic_import import dynamic_import

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--cfg_path", type=str, required=True)
args = parser.parse_args()
# yapf: enable

if __name__ == '__main__':
    args.cfg_path = os.path.abspath(os.path.expanduser(args.cfg_path))
    with open(args.cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    model_conf = config['model']
    data_conf = config['data']
    feat_conf = config['feature']
    scoring_conf = config['scoring']

    # Dataset
    ds_class = dynamic_import(data_conf['dataset'])
    test_ds = ds_class(data_dir=data_conf['data_dir'], mode='test', **feat_conf)
    test_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=scoring_conf['batch_size'], drop_last=False)
    test_loader = paddle.io.DataLoader(
        test_ds,
        batch_sampler=test_sampler,
        num_workers=scoring_conf['num_workers'],
        return_list=True,
        use_buffer_reader=True,
        collate_fn=collate_features, )

    # Model
    backbone_class = dynamic_import(model_conf['backbone'])
    backbone = backbone_class(**model_conf['config'])
    model = KWSModel(backbone=backbone, num_keywords=model_conf['num_keywords'])
    model.set_state_dict(paddle.load(scoring_conf['checkpoint']))
    model.eval()

    with paddle.no_grad(), open(
            scoring_conf['score_file'], 'w', encoding='utf8') as fout:
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

    print('Result saved to: {}'.format(scoring_conf['score_file']))
