"""Prepare Aishell mandarin dataset

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import soundfile
import json
import argparse
from data_utils.utility import download, unpack

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

URL_ROOT = 'http://www.openslr.org/resources/33'
DATA_URL = URL_ROOT + '/data_aishell.tgz'
MD5_DATA = '2f494334227864a8a8fec932999db9d8'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/Aishell",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []
    transcript_path = os.path.join(data_dir, 'transcript',
                                   'aishell_transcript_v0.8.txt')
    transcript_dict = {}
    for line in codecs.open(transcript_path, 'r', 'utf-8'):
        line = line.strip()
        if line == '': continue
        audio_id, text = line.split(' ', 1)
        # remove withespace
        text = ''.join(text.split())
        transcript_dict[audio_id] = text

    data_types = ['train', 'dev', 'test']
    for type in data_types:
        del json_lines[:]
        audio_dir = os.path.join(data_dir, 'wav', type)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for fname in filelist:
                audio_path = os.path.join(subfolder, fname)
                audio_id = fname[:-4]
                # if no transcription for audio then skipped
                if audio_id not in transcript_dict:
                    continue
                audio_data, samplerate = soundfile.read(audio_path)
                duration = float(len(audio_data) / samplerate)
                text = transcript_dict[audio_id]
                json_lines.append(
                    json.dumps(
                        {
                            'audio_filepath': audio_path,
                            'duration': duration,
                            'text': text
                        },
                        ensure_ascii=False))
        manifest_path = manifest_path_prefix + '.' + type
        with codecs.open(manifest_path, 'w', 'utf-8') as fout:
            for line in json_lines:
                fout.write(line + '\n')


def prepare_dataset(url, md5sum, target_dir, manifest_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'data_aishell')
    if not os.path.exists(data_dir):
        filepath = download(url, md5sum, target_dir)
        unpack(filepath, target_dir)
        # unpack all audio tar files
        audio_dir = os.path.join(data_dir, 'wav')
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for ftar in filelist:
                unpack(os.path.join(subfolder, ftar), subfolder, True)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)
    create_manifest(data_dir, manifest_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        url=DATA_URL,
        md5sum=MD5_DATA,
        target_dir=args.target_dir,
        manifest_path=args.manifest_prefix)


if __name__ == '__main__':
    main()
