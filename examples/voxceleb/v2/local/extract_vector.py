#!/usr/bin/python3
#! coding:utf-8

#!/usr/bin/env python3
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
"""
Extract Speaker Embedding from the model
"""

import os
import yaml
from yacs.config import CfgNode
import argparse
import paddle
from paddlespeech.vector.models.model import build_sid_models
def main(args):
    model_path = args.model + ".pdparams"
    print("model: {}".format(model_path))
    print("data: {}".format(args.data))
    print("config: {}".format(args.config))
    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))
    model = build_sid_models(config)
    model.set_state_dict(paddle.load(model_path))
    model.eval()
    spks_embedding = {}
    vector_feat = paddle.load(args.data)
    for utt_id in vector_feat.keys():
        feat = paddle.to_tensor(vector_feat[utt_id]["wav"]).astype("float32")
        feat = paddle.unsqueeze(feat, axis=0)
        feat = paddle.transpose(feat, perm=[0, 2, 1])
        # print("feat shape: {}".format(feat.shape))
        utt_embedding = model(feat).squeeze(axis=0)
        # utt_embedding = paddle.squeeze(utt_embedding, axis=0)
        utt_embedding = paddle.transpose(utt_embedding, perm=[1, 0])
        spks_embedding.setdefault(vector_feat[utt_id]["spk_id"], []).append(utt_embedding)

    spk_mean_embedding = dict()
    for spk_id in spks_embedding.keys():
        embedding = paddle.cat(spks_embedding[spk_id])
        embedding = paddle.mean(embedding, axis=0)
        # 对向量进行归一化
        # print("embdding shape: {}".format(embedding.shape))
        spk_mean_embedding[spk_id] = embedding.numpy()

    print("store the speaker embedding to {}".format(args.spker_embedding))
    paddle.save(spk_mean_embedding, args.spker_embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="./exp/model", type=str, help="speaker embedding model")
    parser.add_argument("--data", default="./data/dataset.feat", type=str, help="dataset feature")
    parser.add_argument("--config", default="./conf/default.yaml", type=str, help="train config")
    parser.add_argument("--spker-embedding", default="./exp/model/spk_embedding.vector", type=str, help="speaker embedding")

    args = parser.parse_args()
    # paddle.device.set_device("cpu")
    main(args)