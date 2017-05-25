"""
   Download, unpack and create manifest for Librespeech dataset.

   Manifest is a json file with each line containing one audio clip filepath,
   its transcription text string, and its duration. It servers as a unified
   interfance to organize different data sets.
"""

import paddle.v2 as paddle
import os
import wget
import tarfile
import argparse
import soundfile
import json

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

URL_TEST = "http://www.openslr.org/resources/12/test-clean.tar.gz"
URL_DEV = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
URL_TRAIN = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"

parser = argparse.ArgumentParser(
    description='Downloads and prepare LibriSpeech dataset.')
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/Libri",
    type=str,
    help="Directory to save the dataset.")
parser.add_argument(
    "--manifest",
    default="./libri.manifest",
    type=str,
    help="Filepath prefix for output manifests.")
args = parser.parse_args()


def download(url, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filepath = os.path.join(target_dir, url.split("/")[-1])
    if not os.path.exists(filepath):
        print("Downloading %s ..." % url)
        wget.download(url, target_dir)
        print("")
    return filepath


def unpack(filepath, target_dir):
    print("Unpacking %s ..." % filepath)
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()
    return target_dir


def create_manifest(data_dir, manifest_path):
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


def prepare_dataset(url, target_dir, manifest_path):
    filepath = download(url, target_dir)
    unpacked_dir = unpack(filepath, target_dir)
    create_manifest(unpacked_dir, manifest_path)


def main():
    prepare_dataset(
        url=URL_TEST,
        target_dir=os.path.join(args.target_dir),
        manifest_path=args.manifest + ".test")
    prepare_dataset(
        url=URL_DEV,
        target_dir=os.path.join(args.target_dir),
        manifest_path=args.manifest + ".dev")
    prepare_dataset(
        url=URL_TRAIN,
        target_dir=os.path.join(args.target_dir),
        manifest_path=args.manifest + ".train")


if __name__ == '__main__':
    main()
