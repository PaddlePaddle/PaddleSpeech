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
"""Prepare Librispeech ASR datasets.

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
import argparse
import codecs
import json
import os
import re
import string
from pathlib import Path

import soundfile

from utils.utility import unzip

URL_ROOT = ""
MD5_DATA = "45c68037c7fdfe063a43c851f181fb2d"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default='~/.cache/paddle/dataset/speech/timit',
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()

#: A string containing Chinese punctuation marks (non-stops).
non_stops = (
    # Fullwidth ASCII variants
    '\uFF02\uFF03\uFF04\uFF05\uFF06\uFF07\uFF08\uFF09\uFF0A\uFF0B\uFF0C\uFF0D'
    '\uFF0F\uFF1A\uFF1B\uFF1C\uFF1D\uFF1E\uFF20\uFF3B\uFF3C\uFF3D\uFF3E\uFF3F'
    '\uFF40\uFF5B\uFF5C\uFF5D\uFF5E\uFF5F\uFF60'

    # Halfwidth CJK punctuation
    '\uFF62\uFF63\uFF64'

    # CJK symbols and punctuation
    '\u3000\u3001\u3003'

    # CJK angle and corner brackets
    '\u3008\u3009\u300A\u300B\u300C\u300D\u300E\u300F\u3010\u3011'

    # CJK brackets and symbols/punctuation
    '\u3014\u3015\u3016\u3017\u3018\u3019\u301A\u301B\u301C\u301D\u301E\u301F'

    # Other CJK symbols
    '\u3030'

    # Special CJK indicators
    '\u303E\u303F'

    # Dashes
    '\u2013\u2014'

    # Quotation marks and apostrophe
    '\u2018\u2019\u201B\u201C\u201D\u201E\u201F'

    # General punctuation
    '\u2026\u2027'

    # Overscores and underscores
    '\uFE4F'

    # Small form variants
    '\uFE51\uFE54'

    # Latin punctuation
    '\u00B7')

#: A string of Chinese stops.
stops = (
    '\uFF01'  # Fullwidth exclamation mark
    '\uFF1F'  # Fullwidth question mark
    '\uFF61'  # Halfwidth ideographic full stop
    '\u3002'  # Ideographic full stop
)

#: A string containing all Chinese punctuation.
punctuation = non_stops + stops


def tn(text):
    # lower text
    text = text.lower()
    # remove punc
    text = re.sub(f'[{punctuation}{string.punctuation}]', "", text)
    return text


def read_txt(filepath: str) -> str:
    with open(filepath, 'r') as f:
        line = f.read().strip().split(maxsplit=2)[2]
        return tn(line)


def read_algin(filepath: str) -> str:
    """read word or phone alignment file.
    <start-sample> <end-sample> <token><newline>
    
    Args:
        filepath (str): [description]

    Returns:
        str: token sepearte by <space>
    """
    aligns = []  # (start, end, token)
    with open(filepath, 'r') as f:
        for line in f:
            items = line.strip().split()
            # for phone: (Note: beginning and ending silence regions are marked with h#)
            if items[2].strip() == 'h#':
                continue
            aligns.append(items)
    return ' '.join([item[2] for item in aligns])


def create_manifest(data_dir, manifest_path_prefix):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []
    utts = set()

    data_types = ['TRAIN', 'TEST']
    for dtype in data_types:
        del json_lines[:]
        total_sec = 0.0
        total_text = 0.0
        total_num = 0

        audio_dir = Path(os.path.join(data_dir, dtype))
        for fname in sorted(audio_dir.rglob('*.WAV')):
            audio_path = fname.resolve()  # .WAV
            audio_id = audio_path.stem
            # if uttid exits,  then skipped
            if audio_id in utts:
                continue

            utts.add(audio_id)
            text_path = audio_path.with_suffix('.TXT')
            phone_path = audio_path.with_suffix('.PHN')
            word_path = audio_path.with_suffix('.WRD')

            audio_data, samplerate = soundfile.read(
                str(audio_path), dtype='int16')
            duration = float(len(audio_data) / samplerate)
            word_text = read_txt(text_path)
            phone_text = read_algin(phone_path)

            gender_spk = str(audio_path.parent.stem)
            spk = gender_spk[1:]
            gender = gender_spk[0]
            utt_id = '_'.join([spk, gender, audio_id])
            # not dump alignment infos
            json_lines.append(
                json.dumps(
                    {
                        'utt': utt_id,
                        'utt2spk': spk,
                        'utt2gender': gender,
                        'feat': str(audio_path),
                        'feat_shape': (duration, ),  # second
                        'text': word_text,  # word
                        'phone': phone_text,
                    },
                    ensure_ascii=False))

            total_sec += duration
            total_text += len(word_text.split())
            total_num += 1

        manifest_path = manifest_path_prefix + '.' + dtype.lower()
        with codecs.open(manifest_path, 'w', 'utf-8') as fout:
            for line in json_lines:
                fout.write(line + '\n')

        manifest_dir = os.path.dirname(manifest_path_prefix)
        meta_path = os.path.join(manifest_dir, dtype.lower()) + '.meta'
        with open(meta_path, 'w') as f:
            print(f"{dtype}:", file=f)
            print(f"{total_num} utts", file=f)
            print(f"{total_sec / (60*60)} h", file=f)
            print(f"{total_text} text", file=f)
            print(f"{total_text / total_sec} text/sec", file=f)
            print(f"{total_sec / total_num} sec/utt", file=f)


def prepare_dataset(url, md5sum, target_dir, manifest_path):
    """Download, unpack and create summmary manifest file.
    """
    filepath = os.path.join(target_dir, "TIMIT.zip")
    if not os.path.exists(filepath):
        print(f"Please download TIMIT.zip into {target_dir}.")
        raise FileNotFoundError

    if not os.path.exists(os.path.join(target_dir, "TIMIT")):
        # check md5sum
        assert check_md5sum(filepath, md5sum)
        # unpack
        unzip(filepath, target_dir)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)
    # create manifest json file
    create_manifest(os.path.join(target_dir, "TIMIT"), manifest_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(URL_ROOT, MD5_DATA, args.target_dir, args.manifest_prefix)
    print("Data download and manifest prepare done!")


if __name__ == '__main__':
    main()
