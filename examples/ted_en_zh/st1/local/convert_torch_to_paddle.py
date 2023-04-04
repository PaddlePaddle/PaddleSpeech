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

import paddle
import torch

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


def torch2paddle(args):
    paddle.set_device('cpu')
    paddle_model_dict = {}
    torch_model = torch.load(args.torch_ckpt, map_location='cpu')
    cnt = 0
    for k, v in torch_model['model'].items():
        # encoder.embed.* --> encoder.embed.*
        if k.startswith('encoder.embed'):
            if v.ndim == 2:
                v = v.transpose(0, 1)
            paddle_model_dict[k] = v.numpy()
            cnt += 1
            logger.info(
                f"Convert torch weight: {k} to paddlepaddle weight: {k}, shape is {v.shape}"
            )

        # encoder.after_norm.* --> encoder.after_norm.*
        # encoder.after_norm.* --> decoder.after_norm.*
        # encoder.after_norm.* --> st_decoder.after_norm.*
        if k.startswith('encoder.after_norm'):
            paddle_model_dict[k] = v.numpy()
            cnt += 1
            paddle_model_dict[k.replace('en', 'de')] = v.numpy()
            logger.info(
                f"Convert torch weight: {k} to paddlepaddle weight: {k.replace('en','de')}, shape is {v.shape}"
            )
            paddle_model_dict['st_' + k.replace('en', 'de')] = v.numpy()
            logger.info(
                f"Convert torch weight: {k} to paddlepaddle weight: {'st_'+ k.replace('en','de')}, shape is {v.shape}"
            )
            cnt += 2

        # encoder.encoders.* --> encoder.encoders.*
        # encoder.encoders.* (last six layers) --> decoder.encoders.* (first six layers)
        # encoder.encoders.* (last six layers) --> st_decoder.encoders.* (first six layers)
        if k.startswith('encoder.encoders'):
            if v.ndim == 2:
                v = v.transpose(0, 1)
            paddle_model_dict[k] = v.numpy()
            logger.info(
                f"Convert torch weight: {k} to paddlepaddle weight: {k}, shape is {v.shape}"
            )
            cnt += 1
            origin_k = k
            k_split = k.split('.')
            if int(k_split[2]) >= 6:
                k = k.replace(k_split[2], str(int(k_split[2]) - 6))
                paddle_model_dict[k.replace('en', 'de')] = v.numpy()
                logger.info(
                    f"Convert torch weight: {origin_k} to paddlepaddle weight: {k.replace('en','de')}, shape is {v.shape}"
                )
                paddle_model_dict['st_' + k.replace('en', 'de')] = v.numpy()
                logger.info(
                    f"Convert torch weight: {origin_k} to paddlepaddle weight: {'st_'+ k.replace('en','de')}, shape is {v.shape}"
                )
                cnt += 2
    logger.info(f"Convert {cnt} weights totally from torch to paddlepaddle")
    paddle.save(paddle_model_dict, args.paddle_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch_ckpt',
                        type=str,
                        default='/home/snapshot.ep.98',
                        help="Path to torch checkpoint.")
    parser.add_argument('--paddle_ckpt',
                        type=str,
                        default='paddle.98.pdparams',
                        help="Path to save paddlepaddle checkpoint.")
    args = parser.parse_args()
    torch2paddle(args)
