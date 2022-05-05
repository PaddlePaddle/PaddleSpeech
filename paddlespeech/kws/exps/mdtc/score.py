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
import paddle
from tqdm import tqdm
from yacs.config import CfgNode

from paddlespeech.kws.exps.mdtc.collate import collate_features
from paddlespeech.kws.models.mdtc import KWSModel
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.dynamic_import import dynamic_import

if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help='model checkpoint for evaluation.')
    parser.add_argument(
        "--score_file",
        type=str,
        default='./scores.txt',
        help='output file of trigger scores')
    args = parser.parse_args()

    # https://yaml.org/type/float.html
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)

    # Dataset
    ds_class = dynamic_import(config['dataset'])
    test_ds = ds_class(
        data_dir=config['data_dir'],
        mode='test',
        feat_type=config['feat_type'],
        sample_rate=config['sample_rate'],
        frame_shift=config['frame_shift'],
        frame_length=config['frame_length'],
        n_mels=config['n_mels'], )
    test_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=config['batch_size'], drop_last=False)
    test_loader = paddle.io.DataLoader(
        test_ds,
        batch_sampler=test_sampler,
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
    model.set_state_dict(paddle.load(args.ckpt))
    model.eval()

    with paddle.no_grad(), open(args.score_file, 'w', encoding='utf8') as f:
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
                    f.write('{} {} {}\n'.format(key, keyword_i, score_frames))

    print('Result saved to: {}'.format(args.score_file))
