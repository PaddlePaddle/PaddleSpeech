"""Prepare Librispeech ASR datasets.

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.util
import os
import sys
import argparse
import soundfile
import json
import codecs
from data_utils.utility import download, unpack

URL_ROOT = "http://www.openslr.org/resources/12"
URL_TEST_CLEAN = URL_ROOT + "/test-clean.tar.gz"
URL_TEST_OTHER = URL_ROOT + "/test-other.tar.gz"
URL_DEV_CLEAN = URL_ROOT + "/dev-clean.tar.gz"
URL_DEV_OTHER = URL_ROOT + "/dev-other.tar.gz"
URL_TRAIN_CLEAN_100 = URL_ROOT + "/train-clean-100.tar.gz"
URL_TRAIN_CLEAN_360 = URL_ROOT + "/train-clean-360.tar.gz"
URL_TRAIN_OTHER_500 = URL_ROOT + "/train-other-500.tar.gz"

MD5_TEST_CLEAN = "32fa31d27d2e1cad72775fee3f4849a9"
MD5_TEST_OTHER = "fb5a50374b501bb3bac4815ee91d3135"
MD5_DEV_CLEAN = "42e2234ba48799c1f50f24a7926300a1"
MD5_DEV_OTHER = "c8d0bcc9cca99d4f8b62fcc847357931"
MD5_TRAIN_CLEAN_100 = "2a93770f6d5c6c964bc36631d331a522"
MD5_TRAIN_CLEAN_360 = "c0e676e450a7ff2f54aeade5171606fa"
MD5_TRAIN_OTHER_500 = "d1a0fd59409feb2c614ce4d30c387708"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default='~/.cache/paddle/dataset/speech/libri',
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
parser.add_argument(
    "--full_download",
    default="True",
    type=distutils.util.strtobool,
    help="Download all datasets for Librispeech."
    " If False, only download a minimal requirement (test-clean, dev-clean"
    " train-clean-100). (default: %(default)s)")
args = parser.parse_args()


def create_manifest(data_dir, manifest_path):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path)
    json_lines = []
    for subfolder, _, filelist in sorted(os.walk(data_dir)):
        text_filelist = [
            filename for filename in filelist if filename.endswith('trans.txt')
        ]
        if len(text_filelist) > 0:
            text_filepath = os.path.join(data_dir, subfolder, text_filelist[0])
            for line in open(text_filepath):
                segments = line.strip().split()
                text = ' '.join(segments[1:]).lower()
                audio_filepath = os.path.join(data_dir, subfolder,
                                              segments[0] + '.flac')
                audio_data, samplerate = soundfile.read(audio_filepath)
                duration = float(len(audio_data)) / samplerate
                json_lines.append(
                    json.dumps({
                        'audio_filepath': audio_filepath,
                        'duration': duration,
                        'text': text
                    }))
    with codecs.open(manifest_path, 'w', 'utf-8') as out_file:
        for line in json_lines:
            out_file.write(line + '\n')


def prepare_dataset(url, md5sum, target_dir, manifest_path):
    """Download, unpack and create summmary manifest file.
    """
    if not os.path.exists(os.path.join(target_dir, "LibriSpeech")):
        # download
        filepath = download(url, md5sum, target_dir)
        # unpack
        unpack(filepath, target_dir)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)
    # create manifest json file
    create_manifest(target_dir, manifest_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        url=URL_TEST_CLEAN,
        md5sum=MD5_TEST_CLEAN,
        target_dir=os.path.join(args.target_dir, "test-clean"),
        manifest_path=args.manifest_prefix + ".test-clean")
    prepare_dataset(
        url=URL_DEV_CLEAN,
        md5sum=MD5_DEV_CLEAN,
        target_dir=os.path.join(args.target_dir, "dev-clean"),
        manifest_path=args.manifest_prefix + ".dev-clean")
    if args.full_download:
        prepare_dataset(
            url=URL_TRAIN_CLEAN_100,
            md5sum=MD5_TRAIN_CLEAN_100,
            target_dir=os.path.join(args.target_dir, "train-clean-100"),
            manifest_path=args.manifest_prefix + ".train-clean-100")
        prepare_dataset(
            url=URL_TEST_OTHER,
            md5sum=MD5_TEST_OTHER,
            target_dir=os.path.join(args.target_dir, "test-other"),
            manifest_path=args.manifest_prefix + ".test-other")
        prepare_dataset(
            url=URL_DEV_OTHER,
            md5sum=MD5_DEV_OTHER,
            target_dir=os.path.join(args.target_dir, "dev-other"),
            manifest_path=args.manifest_prefix + ".dev-other")
        prepare_dataset(
            url=URL_TRAIN_CLEAN_360,
            md5sum=MD5_TRAIN_CLEAN_360,
            target_dir=os.path.join(args.target_dir, "train-clean-360"),
            manifest_path=args.manifest_prefix + ".train-clean-360")
        prepare_dataset(
            url=URL_TRAIN_OTHER_500,
            md5sum=MD5_TRAIN_OTHER_500,
            target_dir=os.path.join(args.target_dir, "train-other-500"),
            manifest_path=args.manifest_prefix + ".train-other-500")


if __name__ == '__main__':
    main()
