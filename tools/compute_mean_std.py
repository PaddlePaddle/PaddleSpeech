"""Compute mean and std for feature normalizer, and save to file."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import _init_paths
from data_utils.normalizer import FeatureNormalizer
from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.audio_featurizer import AudioFeaturizer

parser = argparse.ArgumentParser(
    description='Computing mean and stddev for feature normalizer.')
parser.add_argument(
    "--specgram_type",
    default='linear',
    type=str,
    help="Feature type of audio data: 'linear' (power spectrum)"
    " or 'mfcc'. (default: %(default)s)")
parser.add_argument(
    "--manifest_path",
    default='datasets/manifest.train',
    type=str,
    help="Manifest path for computing normalizer's mean and stddev."
    "(default: %(default)s)")
parser.add_argument(
    "--num_samples",
    default=2000,
    type=int,
    help="Number of samples for computing mean and stddev. "
    "(default: %(default)s)")
parser.add_argument(
    "--augmentation_config",
    default='{}',
    type=str,
    help="Augmentation configuration in json-format. "
    "(default: %(default)s)")
parser.add_argument(
    "--output_file",
    default='mean_std.npz',
    type=str,
    help="Filepath to write mean and std to (.npz)."
    "(default: %(default)s)")
args = parser.parse_args()


def main():
    augmentation_pipeline = AugmentationPipeline(args.augmentation_config)
    audio_featurizer = AudioFeaturizer(specgram_type=args.specgram_type)

    def augment_and_featurize(audio_segment):
        augmentation_pipeline.transform_audio(audio_segment)
        return audio_featurizer.featurize(audio_segment)

    normalizer = FeatureNormalizer(
        mean_std_filepath=None,
        manifest_path=args.manifest_path,
        featurize_func=augment_and_featurize,
        num_samples=args.num_samples)
    normalizer.write_to_file(args.output_file)


if __name__ == '__main__':
    main()
