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
"""Prepare VoxForge dataset

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
import argparse
import codecs
import datetime
import json
import os
import shutil
import subprocess

import soundfile

from utils.utility import download_multi
from utils.utility import getfile_insensitive
from utils.utility import unpack

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

DATA_URL = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/' \
           'Audio/Main/16kHz_16bit'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/VoxForge",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--dialects",
    default=[
        'american', 'british', 'australian', 'european', 'irish', 'canadian',
        'indian'
    ],
    nargs='+',
    type=str,
    help="Dialect types. (default: %(default)s)")
parser.add_argument(
    "--is_merge_dialect",
    default=True,
    type=bool,
    help="If set True, manifests of american dialect and canadian dialect will "
    "be merged to american-canadian dialect; manifests of british "
    "dialect, irish dialect and australian dialect will be merged to "
    "commonwealth dialect. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def download_and_unpack(target_dir, url):
    wget_args = '-q -l 1 -N -nd -c -e robots=off -A tgz -r -np'
    tgz_dir = os.path.join(target_dir, 'tgz')
    exit_code = download_multi(url, tgz_dir, wget_args)
    if exit_code != 0:
        print('Download tgz audio files failed with exit code %d.' % exit_code)
    else:
        print('Download done, start unpacking ...')
        audio_dir = os.path.join(target_dir, 'audio')
        for root, dirs, files in os.walk(tgz_dir):
            for file in files:
                print(file)
                if file.endswith('.tgz'):
                    unpack(os.path.join(root, file), audio_dir)


def select_dialects(target_dir, dialect_list):
    """Classify audio files by dialect."""
    dialect_root_dir = os.path.join(target_dir, 'dialect')
    if os.path.exists(dialect_root_dir):
        shutil.rmtree(dialect_root_dir)
    os.mkdir(dialect_root_dir)
    audio_dir = os.path.abspath(os.path.join(target_dir, 'audio'))
    for dialect in dialect_list:
        # filter files by dialect
        command = 'find %s -iwholename "*etc/readme*" -exec egrep -iHl \
            "pronunciation dialect.*%s" {} \;' % (audio_dir, dialect)
        p = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        dialect_dir = os.path.join(dialect_root_dir, dialect)
        if os.path.exists(dialect_dir):
            shutil.rmtree(dialect_dir)
        os.mkdir(dialect_dir)
        for path in output.splitlines():
            src_dir = os.path.dirname(os.path.dirname(path))
            link = os.path.basename(os.path.normpath(src_dir))
            os.symlink(src_dir, os.path.join(dialect_dir, link))


def generate_manifest(data_dir, manifest_path):
    json_lines = []

    for path in os.listdir(data_dir):
        audio_link = os.path.join(data_dir, path)
        assert os.path.islink(
            audio_link), '%s should be symbolic link.' % audio_link
        actual_audio_dir = os.path.abspath(os.readlink(audio_link))

        audio_type = ''
        if os.path.isdir(os.path.join(actual_audio_dir, 'wav')):
            audio_type = 'wav'
        elif os.path.isdir(os.path.join(actual_audio_dir, 'flac')):
            audio_type = 'flac'
        else:
            print('Unknown audio type, skipped processing %s.' %
                  actual_audio_dir)
            continue

        etc_dir = os.path.join(actual_audio_dir, 'etc')
        prompts_file = os.path.join(etc_dir, 'PROMPTS')
        if not os.path.isfile(prompts_file):
            print('PROMPTS file missing, skip processing %s.' %
                  actual_audio_dir)
            continue

        readme_file = getfile_insensitive(os.path.join(etc_dir, 'README'))
        if readme_file is None:
            print('README file missing, skip processing %s.' % actual_audio_dir)
            continue

        for line in file(prompts_file):
            u, trans = line.strip().split(None, 1)
            u_parts = u.split('/')

            # try to format the date time
            try:
                speaker, date, sfx = u_parts[-3].split('-')
                obj = datetime.datetime.strptime(date, '%y.%m.%d')
                formatted = obj.strftime('%Y%m%d')
                u_parts[-3] = '-'.join([speaker, formatted, sfx])
            except Exception as e:
                pass

            if len(u_parts) < 2:
                u_parts = [audio_type] + u_parts
            u_parts[-2] = audio_type
            u_parts[-1] += '.' + audio_type
            u = os.path.join(actual_audio_dir, '/'.join(u_parts[-2:]))

            if not os.path.isfile(u):
                print('Audio file missing, skip processing %s.' % u)
                continue

            if os.stat(u).st_size == 0:
                print('Empty audio file, skip processing %s.' % u)
                continue

            trans = trans.strip().replace('-', ' ')
            if not trans.isupper() or \
                not trans.strip().replace(' ', '').replace("'", "").isalpha():
                print("Transcript not normalized properly, skip processing %s."
                      % u)
                continue

            audio_data, samplerate = soundfile.read(u)
            duration = float(len(audio_data)) / samplerate

            utt = os.path.splitext(os.path.basename(u))[0]
            json_lines.append(
                json.dumps({
                    'utt': utt,
                    'utt2spk': speaker,
                    'feat': u,
                    'feat_shape': (duration, ),  #second
                    'text': trans.lower()
                }))

    with codecs.open(manifest_path, 'w', 'utf-8') as fout:
        for line in json_lines:
            fout.write(line + '\n')


def merge_manifests(manifest_files, save_path):
    lines = []
    for manifest_file in manifest_files:
        line = codecs.open(manifest_file, 'r', 'utf-8').readlines()
        lines += line

    with codecs.open(save_path, 'w', 'utf-8') as fout:
        for line in lines:
            fout.write(line)


def prepare_dataset(url, dialects, target_dir, manifest_prefix, is_merge):
    download_and_unpack(target_dir, url)
    select_dialects(target_dir, dialects)
    american_canadian_manifests = []
    commonwealth_manifests = []
    for dialect in dialects:
        dialect_dir = os.path.join(target_dir, 'dialect', dialect)
        manifest_fpath = manifest_prefix + '.' + dialect
        if dialect == 'american' or dialect == 'canadian':
            american_canadian_manifests.append(manifest_fpath)
        if dialect == 'australian' \
                or dialect == 'british' \
                or dialect == 'irish':
            commonwealth_manifests.append(manifest_fpath)
        generate_manifest(dialect_dir, manifest_fpath)

    if is_merge:
        if len(american_canadian_manifests) > 0:
            manifest_fpath = manifest_prefix + '.american-canadian'
            merge_manifests(american_canadian_manifests, manifest_fpath)
        if len(commonwealth_manifests) > 0:
            manifest_fpath = manifest_prefix + '.commonwealth'
            merge_manifests(commonwealth_manifests, manifest_fpath)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(DATA_URL, args.dialects, args.target_dir,
                    args.manifest_prefix, args.is_merge_dialect)


if __name__ == '__main__':
    main()
