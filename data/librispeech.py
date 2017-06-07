"""
    Download, unpack and create manifest file for the Librespeech dataset.

    A manifest file is a dataset summarization, with each line a json format
    string containing meta data for one audio clip, including its filepath,
    transcription string, and duration. It serves as a unified interface for
    different data sets.
"""

import paddle.v2 as paddle
from paddle.v2.dataset.common import md5file
import distutils.util
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
MD5_TEST_OTHER = "fb5a50374b501bb3bac4815ee91d3135"
MD5_DEV_CLEAN = "42e2234ba48799c1f50f24a7926300a1"
MD5_DEV_OTHER = "c8d0bcc9cca99d4f8b62fcc847357931"
MD5_TRAIN_CLEAN_100 = "2a93770f6d5c6c964bc36631d331a522"
MD5_TRAIN_CLEAN_360 = "c0e676e450a7ff2f54aeade5171606fa"
MD5_TRAIN_OTHER_500 = "d1a0fd59409feb2c614ce4d30c387708"

NUM_LINES_TEST_CLEAN = 2620
NUM_LINES_TEST_OTHER = 2939
NUM_LINES_DEV_CLEAN = 2703
NUM_LINES_DEV_OTHER = 2864
NUM_LINES_TRAIN_CLEAN_100 = 28539
NUM_LINES_TRAIN_CLEAN_360 = 104014
NUM_LINES_TRAIN_OTHER_500 = 148688

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
parser.add_argument(
    "--full_download",
    default="True",
    type=distutils.util.strtobool,
    help="Download all datasets for Librispeech."
    " If False, only download a minimal requirement (test-clean, dev-clean"
    " train-clean-100). (default: %(default)s)")
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
        if not md5file(filepath) == md5sum:
            raise RuntimeError("MD5 checksum failed.")
    else:
        print("File exists, skip downloading. (%s)" % filepath)
    return filepath


def unpack(filepath, target_dir):
    """
    Unpack the file to the target_dir.
    """
    print("Unpacking %s ..." % filepath)
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()


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
    with open(manifest_path, 'w') as out_file:
        for line in json_lines:
            out_file.write(line + '\n')


def verify_file_line_number(filepath, num_lines):
    with open(filepath, 'r') as file:
        return len(file.readlines()) == num_lines


def prepare_dataset(url, md5sum, target_dir, manifest_path, num_lines):
    """
    Download, unpack and create summmary manifest file.
    """
    # download
    filepath = download(url, md5sum, target_dir)
    # unpack
    if not os.path.exists(os.path.join(target_dir, "LibriSpeech")):
        unpack(filepath, target_dir)
    else:
        print("Unpacked data exists, skip unpacking.")
    # create manifest and verify line number
    create_manifest(target_dir, manifest_path)
    if not verify_file_line_number(manifest_path, num_lines):
        raise RuntimeError("Manifest line number check failed. "
                           "Please remove directory and try running the script "
                           "again.")


def main():
    prepare_dataset(
        url=URL_TEST_CLEAN,
        md5sum=MD5_TEST_CLEAN,
        target_dir=os.path.join(args.target_dir, "test-clean"),
        manifest_path=args.manifest_prefix + ".test-clean",
        num_lines=NUM_LINES_TEST_CLEAN)
    prepare_dataset(
        url=URL_DEV_CLEAN,
        md5sum=MD5_DEV_CLEAN,
        target_dir=os.path.join(args.target_dir, "dev-clean"),
        manifest_path=args.manifest_prefix + ".dev-clean",
        num_lines=NUM_LINES_DEV_CLEAN)
    prepare_dataset(
        url=URL_TRAIN_CLEAN_100,
        md5sum=MD5_TRAIN_CLEAN_100,
        target_dir=os.path.join(args.target_dir, "train-clean-100"),
        manifest_path=args.manifest_prefix + ".train-clean-100",
        num_lines=NUM_LINES_TRAIN_CLEAN_100)
    if args.full_download:
        prepare_dataset(
            url=URL_TEST_OTHER,
            md5sum=MD5_TEST_OTHER,
            target_dir=os.path.join(args.target_dir, "test-other"),
            manifest_path=args.manifest_prefix + ".test-other",
            num_lines=NUM_LINES_TEST_OTHER)
        prepare_dataset(
            url=URL_DEV_OTHER,
            md5sum=MD5_DEV_OTHER,
            target_dir=os.path.join(args.target_dir, "dev-other"),
            manifest_path=args.manifest_prefix + ".dev-other",
            num_lines=NUM_LINES_DEV_OTHER)
        prepare_dataset(
            url=URL_TRAIN_CLEAN_360,
            md5sum=MD5_TRAIN_CLEAN_360,
            target_dir=os.path.join(args.target_dir, "train-clean-360"),
            manifest_path=args.manifest_prefix + ".train-clean-360",
            num_lines=NUM_LINES_TRAIN_CLEAN_360)
        prepare_dataset(
            url=URL_TRAIN_OTHER_500,
            md5sum=MD5_TRAIN_OTHER_500,
            target_dir=os.path.join(args.target_dir, "train-other-500"),
            manifest_path=args.manifest_prefix + ".train-other-500",
            num_lines=NUM_LINES_TRAIN_OTHER_500)


if __name__ == '__main__':
    main()
