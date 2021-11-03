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
import codecs
import collections
import json
import os
from typing import Dict

from paddle.io import Dataset
from tqdm import tqdm

from ..backends import load as load_audio
from ..utils.download import decompress
from ..utils.download import download_and_decompress
from ..utils.env import DATA_HOME
from ..utils.log import logger
from .dataset import feat_funcs

__all__ = ['AISHELL1']


class AISHELL1(Dataset):
    """
    This Open Source Mandarin Speech Corpus, AISHELL-ASR0009-OS1, is 178 hours long.
    It is a part of AISHELL-ASR0009, of which utterance contains 11 domains, including
    smart home, autonomous driving, and industrial production. The whole recording was
    put in quiet indoor environment, using 3 different devices at the same time: high
    fidelity microphone (44.1kHz, 16-bit,); Android-system mobile phone (16kHz, 16-bit),
    iOS-system mobile phone (16kHz, 16-bit). Audios in high fidelity were re-sampled
    to 16kHz to build AISHELL- ASR0009-OS1. 400 speakers from different accent areas
    in China were invited to participate in the recording. The manual transcription
    accuracy rate is above 95%, through professional speech annotation and strict
    quality inspection. The corpus is divided into training, development and testing
    sets.

    Reference:
        AISHELL-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline
        https://arxiv.org/abs/1709.05522
    """

    archieves = [
        {
            'url': 'http://www.openslr.org/resources/33/data_aishell.tgz',
            'md5': '2f494334227864a8a8fec932999db9d8',
        },
    ]
    text_meta = os.path.join('data_aishell', 'transcript',
                             'aishell_transcript_v0.8.txt')
    utt_info = collections.namedtuple('META_INFO',
                                      ('file_path', 'utt_id', 'text'))
    audio_path = os.path.join('data_aishell', 'wav')
    manifest_path = os.path.join('data_aishell', 'manifest')
    subset = ['train', 'dev', 'test']

    def __init__(self, subset: str='train', feat_type: str='raw', **kwargs):
        assert subset in self.subset, 'Dataset subset must be one in {}, but got {}'.format(
            self.subset, subset)
        self.subset = subset
        self.feat_type = feat_type
        self.feat_config = kwargs
        self._data = self._get_data()
        super(AISHELL1, self).__init__()

    def _get_text_info(self) -> Dict[str, str]:
        ret = {}
        with open(os.path.join(DATA_HOME, self.text_meta), 'r') as rf:
            for line in rf.readlines()[1:]:
                utt_id, text = map(str.strip, line.split(' ',
                                                         1))  # utt_id, text
                ret.update({utt_id: ''.join(text.split())})
        return ret

    def _get_data(self):
        if not os.path.isdir(os.path.join(DATA_HOME, self.audio_path)) or \
            not os.path.isfile(os.path.join(DATA_HOME, self.text_meta)):
            download_and_decompress(self.archieves, DATA_HOME)
            # Extract *wav from *.tar.gz.
            for root, _, files in os.walk(
                    os.path.join(DATA_HOME, self.audio_path)):
                for file in files:
                    if file.endswith('.tar.gz'):
                        decompress(os.path.join(root, file))
                        os.remove(os.path.join(root, file))

        text_info = self._get_text_info()

        data = []
        for root, _, files in os.walk(
                os.path.join(DATA_HOME, self.audio_path, self.subset)):
            for file in files:
                if file.endswith('.wav'):
                    utt_id = os.path.splitext(file)[0]
                    if utt_id not in text_info:  # There are some utt_id that without label
                        continue
                    text = text_info[utt_id]
                    file_path = os.path.join(root, file)
                    data.append(self.utt_info(file_path, utt_id, text))

        return data

    def _convert_to_record(self, idx: int):
        sample = self._data[idx]

        record = {}
        # To show all fields in a namedtuple: `type(sample)._fields`
        for field in type(sample)._fields:
            record[field] = getattr(sample, field)

        waveform, sr = load_audio(
            sample[0])  # The first element of sample is file path
        feat_func = feat_funcs[self.feat_type]
        feat = feat_func(
            waveform, sample_rate=sr,
            **self.feat_config) if feat_func else waveform
        record.update({'feat': feat, 'duration': len(waveform) / sr})
        return record

    def create_manifest(self, prefix='manifest'):
        if not os.path.isdir(os.path.join(DATA_HOME, self.manifest_path)):
            os.makedirs(os.path.join(DATA_HOME, self.manifest_path))

        manifest_file = os.path.join(DATA_HOME, self.manifest_path,
                                     f'{prefix}.{self.subset}')
        with codecs.open(manifest_file, 'w', 'utf-8') as f:
            for idx in tqdm(range(len(self))):
                record = self._convert_to_record(idx)
                record_line = json.dumps(
                    {
                        'utt': record['utt_id'],
                        'feat': record['file_path'],
                        'feat_shape': (record['duration'], ),
                        'text': record['text']
                    },
                    ensure_ascii=False)
                f.write(record_line + '\n')
        logger.info(f'Manifest file {manifest_file} created.')

    def __getitem__(self, idx):
        record = self._convert_to_record(idx)
        return tuple(record.values())

    def __len__(self):
        return len(self._data)
