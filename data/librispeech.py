"""
   Download, unpack and create manifest for Librespeech dataset.

   Manifest is a json file with each line containing one audio clip filepath,
   its transcription text string, and its duration. It servers as a unified
   interfance to organize different data sets.
"""

import paddle.v2 as paddle
from paddle.v2.dataset.common import md5file
import os
import wget
import tarfile
import argparse
import soundfile
import json

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

URL_ROOT = "http://www.openslr.org/resources/12"
URL_TEST_CLEAN = URL_ROOT + "/test-clean.tar.gz"
URL_TEST_OTHER = URL_ROOT + "/test-other.tar.gz"
URL_DEV_CLEAN = URL_ROOT + "/dev-clean.tar.gz"
URL_DEV_OTHER = URL_ROOT + "/dev-other.tar.gz"
URL_TRAIN_CLEAN_100 = URL_ROOT + "/train-clean-100.tar.gz"
URL_TRAIN_CLEAN_360 = URL_ROOT + "/train-clean-360.tar.gz"
URL_TRAIN_OTHER_500 = URL_ROOT + "/train-other-500.tar.gz"

MD5_TEST_CLEAN = "32fa31d27d2e1cad72775fee3f4849a9"
MD5_DEV_CLEAN = "42e2234ba48799c1f50f24a7926300a1"
MD5_TRAIN_CLEAN_100 = "2a93770f6d5c6c964bc36631d331a522"
MD5_TRAIN_CLEAN_360 = "c0e676e450a7ff2f54aeade5171606fa"
MD5_TRAIN_OTHER_500 = "d1a0fd59409feb2c614ce4d30c387708"

parser = argparse.ArgumentParser(
    description='Downloads and prepare LibriSpeech dataset.')
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/Libri",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest.libri",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def download(url, md5sum, target_dir):
    """
    Download file from url to target_dir, and check md5sum.
    """
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    filepath = os.path.join(target_dir, url.split("/")[-1])
    if not (os.path.exists(filepath) and md5file(filepath) == md5sum):
        print("Downloading %s ..." % url)
        wget.download(url, target_dir)
        print("\nMD5 Chesksum %s ..." % filepath)
        assert md5file(filepath) == md5sum, "MD5 checksum failed."
    return filepath


def unpack(filepath, target_dir):
    """
    Unpack the file to the target_dir.
    """
    print("Unpacking %s ..." % filepath)
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()
    return target_dir


def create_manifest(data_dir, manifest_path):
    """
    Create a manifest file summarizing the dataset (list of filepath and meta
    data).

    Each line of the manifest contains one audio clip filepath, its
    transcription text string, and its duration. Manifest file servers as a
    unified interfance to organize data sets.
    """
    print("Creating manifest %s ..." % manifest_path)
    json_lines = []
    for subfolder, _, filelist in os.walk(data_dir):
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
    with open(manifest_path, 'w') as out_file:
        for line in json_lines:
            out_file.write(line + '\n')


def prepare_dataset(url, md5sum, target_dir, manifest_path):
    """
    Download, unpack and create summmary manifest file.
    """
    filepath = download(url, md5sum, target_dir)
    unpacked_dir = unpack(filepath, target_dir)
    create_manifest(unpacked_dir, manifest_path)


def main():
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
    prepare_dataset(
        url=URL_TRAIN_CLEAN_100,
        md5sum=MD5_TRAIN_CLEAN_100,
        target_dir=os.path.join(args.target_dir, "train-clean-100"),
        manifest_path=args.manifest_prefix + ".train-clean-100")


if __name__ == '__main__':
    main()
