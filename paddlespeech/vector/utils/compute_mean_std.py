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
"""Compute mean and std for feature normalizer, and save to file."""
import argparse
import functools

from paddlespeech.s2t.frontend.augmentor.augmentation import AugmentationPipeline
from paddlespeech.s2t.frontend.featurizer.audio_featurizer import AudioFeaturizer
from paddlespeech.s2t.frontend.normalizer import FeatureNormalizer
from paddlespeech.s2t.utils.utility import add_arguments
from paddlespeech.s2t.utils.utility import print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_samples',      int,    2000,    "# of samples to for statistics.")

add_arg('spectrum_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc, fbank.",
        choices=['linear', 'mfcc', 'fbank'])
add_arg('feat_dim',    int, 13, "Audio feature dim.")
add_arg('delta_delta', bool,  False, "Audio feature with delta delta.")
add_arg('stride_ms', int, 10,  "stride length in ms.")
add_arg('window_ms', int, 20,  "stride length in ms.")
add_arg('sample_rate',  int, 16000,  "target sample rate.")
add_arg('use_dB_normalization', bool, True, "do dB normalization.")
add_arg('target_dB',   int, -20,  "target dB.")

add_arg('manifest_path',    str,
        'data/librispeech/manifest.train',
        "Filepath of manifest to compute normalizer's mean and stddev.")
add_arg('num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for processing')
add_arg('output_path',    str,
        'data/librispeech/mean_std.npz',
        "Filepath of write mean and stddev to (.npz).")
# yapf: disable
args = parser.parse_args()


def main():
    print_arguments(args, globals())

    augmentation_pipeline = AugmentationPipeline('{}')
    audio_featurizer = AudioFeaturizer(
        spectrum_type=args.spectrum_type,
        feat_dim=args.feat_dim,
        delta_delta=args.delta_delta,
        stride_ms=float(args.stride_ms),
        window_ms=float(args.window_ms),
        n_fft=None,
        max_freq=None,
        target_sample_rate=args.sample_rate,
        use_dB_normalization=args.use_dB_normalization,
        target_dB=args.target_dB,
        dither=0.0)

    def augment_and_featurize(audio_segment):
        augmentation_pipeline.transform_audio(audio_segment)
        return audio_featurizer.featurize(audio_segment)

    normalizer = FeatureNormalizer(
        mean_std_filepath=None,
        manifest_path=args.manifest_path,
        featurize_func=augment_and_featurize,
        num_samples=args.num_samples,
        num_workers=args.num_workers)
    normalizer.write_to_file(args.output_path)


if __name__ == '__main__':
    main()
