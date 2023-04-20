# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Modified from Fairseq 2023 (https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_hubert_feature.py)
import logging
import os
import sys
import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import paddle
import soundfile as sf
import tqdm
from feature_utils import dump_feature
from feature_utils import get_path_iterator
from paddlespeech.s2t.models.hubert.hubert_ASR import HubertASR
from paddlespeech.s2t.models.hubert.modules.hubert_model import HubertConfig
from paddlespeech.s2t.models.hubert.modules.hubert_model import HubertModel
from paddlespeech.s2t.models.hubert.modules.hubert_model import HubertPretrainingConfig
from paddlespeech.s2t.modules.align import LayerNorm
from paddlespeech.s2t.utils.utility import UpdateConfig
from yacs.config import CfgNode

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout, )
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    def __init__(self, config, layer, max_chunk=1600000):
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(config)
        self.config.output_dim = 5002
        # import pdb
        # pdb.set_trace()
        model = HubertASR.from_config(self.config)
        # model_dict = paddle.load(self.config.hubert_params_path)
        # model.hubert.set_state_dict(model_dict)

        self.model = model
        with open(self.config.vocab_filepath) as f:
            dicts = [symbol.strip() for symbol in f.readlines()]
        task_cfg = self.model.merge_with_parent(HubertPretrainingConfig,
                                                dict(self.config.task_cfg))
        model_cfg = self.model.merge_with_parent(HubertConfig,
                                                 dict(self.config.model_cfg))
        self.hubert = HubertModel(model_cfg, task_cfg, dicts)
        model_dict = paddle.load(self.config.hubert_params_path)
        self.hubert.set_state_dict(model_dict)

        self.model.eval()
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav, _ = sf.read(path, dtype="float32", always_2d=True)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with paddle.no_grad():
            x = paddle.to_tensor(x).float().cuda()
            # if self.task.cfg.normalize:
            #     x = LayerNorm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.shape[0], self.max_chunk):
                x_chunk = x[:, start:start + self.max_chunk]
                feat_chunk, _ = self.hubert.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer, )
                feat.append(feat_chunk)
        return paddle.concat(feat, 1).squeeze(0)


def main(tsv_dir, split, config, layer, nshard, rank, feat_dir, max_chunk):
    reader = HubertFeatureReader(config, layer, max_chunk)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("config")
    parser.add_argument("layer", type=int)
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
