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
import _init_paths
from data_utils.normalizer import FeatureNormalizer
from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.audio_featurizer import AudioFeaturizer
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_samples',      int,    2000,    "# of samples to for statistics.")
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
add_arg('manifest_path',    str,
        'data/librispeech/manifest.train',
        "Filepath of manifest to compute normalizer's mean and stddev.")
add_arg('output_path',    str,
        'data/librispeech/mean_std.npz',
        "Filepath of write mean and stddev to (.npz).")
# yapf: disable
args = parser.parse_args()


def main():
    print_arguments(args)

    augmentation_pipeline = AugmentationPipeline('{}')
    audio_featurizer = AudioFeaturizer(specgram_type=args.specgram_type)

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
