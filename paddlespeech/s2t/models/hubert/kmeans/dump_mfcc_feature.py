# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Modified from Fairseq 2023 (https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/dump_mfcc_feature.py)
import logging
import os
import sys

import paddle
import paddleaudio
import soundfile as sf
from feature_utils import dump_feature
from feature_utils import get_path_iterator
from python_speech_features import delta

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout, )
logger = logging.getLogger("dump_mfcc_feature")


class MfccFeatureReader(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def read_audio(self, path, ref_len=None):
        wav, _ = sf.read(path, dtype="float32", always_2d=True)
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with paddle.no_grad():
            x = paddle.to_tensor(x, dtype="float32")
            x = x.reshape([1, -1])

            mfccs = paddleaudio.compliance.kaldi.mfcc(
                waveform=x,
                sr=self.sample_rate,
                use_energy=False, )  # (freq, time)

            deltas = delta(mfccs, 2)
            ddeltas = delta(deltas, 2)
            concat = paddle.concat(
                x=[mfccs, paddle.to_tensor(deltas), paddle.to_tensor(ddeltas)],
                axis=-1)
            return concat


def main(tsv_dir, split, nshard, rank, feat_dir, sample_rate):
    reader = MfccFeatureReader(sample_rate)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
