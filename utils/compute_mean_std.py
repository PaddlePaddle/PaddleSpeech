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
from deepspeech.frontend.normalizer import FeatureNormalizer
from deepspeech.frontend.augmentor.augmentation import AugmentationPipeline
from deepspeech.frontend.featurizer.audio_featurizer import AudioFeaturizer
from deepspeech.utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_samples',      int,    2000,    "# of samples to for statistics.")
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc, fbank.",
        choices=['linear', 'mfcc', 'fbank'])
add_arg('feat_dim',    int, 13, "Audio feature dim.")
add_arg('delta_delta',    bool,
        False,
        "Audio feature with delta delta.")
add_arg('stride_ms',    float, 10.0,  "stride length in ms.")
add_arg('window_ms',    float, 20.0,  "stride length in ms.")
add_arg('sample_rate',    int, 16000,  "target sample rate.")
add_arg('manifest_path',    str,
        'data/librispeech/manifest.train',
        "Filepath of manifest to compute normalizer's mean and stddev.")
add_arg('output_path',    str,
        'data/librispeech/mean_std.npz',
        "Filepath of write mean and stddev to (.npz).")
# yapf: disable
args = parser.parse_args()


def main():
    print_arguments(args, globals())

    augmentation_pipeline = AugmentationPipeline('{}')
    audio_featurizer = AudioFeaturizer(
        specgram_type=args.specgram_type,
        feat_dim=args.feat_dim,
        delta_delta=args.delta_delta,
        stride_ms=args.stride_ms,
        window_ms=args.window_ms,
        n_fft=None,
        max_freq=None,
        target_sample_rate=args.sample_rate,
        use_dB_normalization=True,
        target_dB=-20)

    def augment_and_featurize(audio_segment):
        augmentation_pipeline.transform_audio(audio_segment)
        return audio_featurizer.featurize(audio_segment)

    normalizer = FeatureNormalizer(
        mean_std_filepath=None,
        manifest_path=args.manifest_path,
        featurize_func=augment_and_featurize,
        num_samples=args.num_samples)
    normalizer.write_to_file(args.output_path)


if __name__ == '__main__':
    main()
