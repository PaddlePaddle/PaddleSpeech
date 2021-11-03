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
import collections
import os
from typing import List
from typing import Tuple

from ..utils.download import download_and_decompress
from ..utils.env import DATA_HOME
from .dataset import AudioClassificationDataset

__all__ = ['UrbanAcousticScenes', 'UrbanAudioVisualScenes']


class UrbanAcousticScenes(AudioClassificationDataset):
    """
    TAU Urban Acoustic Scenes 2020 Mobile Development dataset contains recordings from
    12 European cities in 10 different acoustic scenes using 4 different devices.
    Additionally, synthetic data for 11 mobile devices was created based on the original
    recordings. Of the 12 cities, two are present only in the evaluation set.

    Reference:
        A multi-device dataset for urban acoustic scene classification
        https://arxiv.org/abs/1807.09840
    """

    source_url = 'https://zenodo.org/record/3819968/files/'
    base_name = 'TAU-urban-acoustic-scenes-2020-mobile-development'
    archieves = [
        {
            'url': source_url + base_name + '.meta.zip',
            'md5': '6eae9db553ce48e4ea246e34e50a3cf5',
        },
        {
            'url': source_url + base_name + '.audio.1.zip',
            'md5': 'b1e85b8a908d3d6a6ab73268f385d5c8',
        },
        {
            'url': source_url + base_name + '.audio.2.zip',
            'md5': '4310a13cc2943d6ce3f70eba7ba4c784',
        },
        {
            'url': source_url + base_name + '.audio.3.zip',
            'md5': 'ed38956c4246abb56190c1e9b602b7b8',
        },
        {
            'url': source_url + base_name + '.audio.4.zip',
            'md5': '97ab8560056b6816808dedc044dcc023',
        },
        {
            'url': source_url + base_name + '.audio.5.zip',
            'md5': 'b50f5e0bfed33cd8e52cb3e7f815c6cb',
        },
        {
            'url': source_url + base_name + '.audio.6.zip',
            'md5': 'fbf856a3a86fff7520549c899dc94372',
        },
        {
            'url': source_url + base_name + '.audio.7.zip',
            'md5': '0dbffe7b6e45564da649378723284062',
        },
        {
            'url': source_url + base_name + '.audio.8.zip',
            'md5': 'bb6f77832bf0bd9f786f965beb251b2e',
        },
        {
            'url': source_url + base_name + '.audio.9.zip',
            'md5': 'a65596a5372eab10c78e08a0de797c9e',
        },
        {
            'url': source_url + base_name + '.audio.10.zip',
            'md5': '2ad595819ffa1d56d2de4c7ed43205a6',
        },
        {
            'url': source_url + base_name + '.audio.11.zip',
            'md5': '0ad29f7040a4e6a22cfd639b3a6738e5',
        },
        {
            'url': source_url + base_name + '.audio.12.zip',
            'md5': 'e5f4400c6b9697295fab4cf507155a2f',
        },
        {
            'url': source_url + base_name + '.audio.13.zip',
            'md5': '8855ab9f9896422746ab4c5d89d8da2f',
        },
        {
            'url': source_url + base_name + '.audio.14.zip',
            'md5': '092ad744452cd3e7de78f988a3d13020',
        },
        {
            'url': source_url + base_name + '.audio.15.zip',
            'md5': '4b5eb85f6592aebf846088d9df76b420',
        },
        {
            'url': source_url + base_name + '.audio.16.zip',
            'md5': '2e0a89723e58a3836be019e6996ae460',
        },
    ]
    label_list = [
        'airport', 'shopping_mall', 'metro_station', 'street_pedestrian',
        'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park'
    ]

    meta = os.path.join(base_name, 'meta.csv')
    meta_info = collections.namedtuple('META_INFO', (
        'filename', 'scene_label', 'identifier', 'source_label'))
    subset_meta = {
        'train': os.path.join(base_name, 'evaluation_setup', 'fold1_train.csv'),
        'dev':
        os.path.join(base_name, 'evaluation_setup', 'fold1_evaluate.csv'),
        'test': os.path.join(base_name, 'evaluation_setup', 'fold1_test.csv'),
    }
    subset_meta_info = collections.namedtuple('SUBSET_META_INFO',
                                              ('filename', 'scene_label'))
    audio_path = os.path.join(base_name, 'audio')

    def __init__(self, mode: str='train', feat_type: str='raw', **kwargs):
        """
        Ags:
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train or dev).
            feat_type (:obj:`str`, `optional`, defaults to `raw`):
                It identifies the feature type that user wants to extrace of an audio file.
        """
        files, labels = self._get_data(mode)
        super(UrbanAcousticScenes, self).__init__(
            files=files, labels=labels, feat_type=feat_type, **kwargs)

    def _get_meta_info(self, subset: str=None,
                       skip_header: bool=True) -> List[collections.namedtuple]:
        if subset is None:
            meta_file = self.meta
            meta_info = self.meta_info
        else:
            assert subset in self.subset_meta, f'Subset must be one in {list(self.subset_meta.keys())}, but got {subset}.'
            meta_file = self.subset_meta[subset]
            meta_info = self.subset_meta_info

        ret = []
        with open(os.path.join(DATA_HOME, meta_file), 'r') as rf:
            lines = rf.readlines()[1:] if skip_header else rf.readlines()
            for line in lines:
                ret.append(meta_info(*line.strip().split('\t')))
        return ret

    def _get_data(self, mode: str) -> Tuple[List[str], List[int]]:
        if not os.path.isdir(os.path.join(DATA_HOME, self.audio_path)) or \
            not os.path.isfile(os.path.join(DATA_HOME, self.meta)):
            download_and_decompress(self.archieves, DATA_HOME)

        meta_info = self._get_meta_info(subset=mode, skip_header=True)

        files = []
        labels = []
        for sample in meta_info:
            filename, label = sample[:2]
            filename = os.path.basename(filename)
            target = self.label_list.index(label)

            files.append(os.path.join(DATA_HOME, self.audio_path, filename))
            labels.append(int(target))

        return files, labels


class UrbanAudioVisualScenes(AudioClassificationDataset):
    """
    TAU Urban Audio Visual Scenes 2021 Development dataset contains synchronized audio
    and video recordings from 12 European cities in 10 different scenes.
    This dataset consists of 10-seconds audio and video segments from 10
    acoustic scenes. The total amount of audio in the development set is 34 hours.

    Reference:
        A Curated Dataset of Urban Scenes for Audio-Visual Scene Analysis
        https://arxiv.org/abs/2011.00030
    """

    source_url = 'https://zenodo.org/record/4477542/files/'
    base_name = 'TAU-urban-audio-visual-scenes-2021-development'

    archieves = [
        {
            'url': source_url + base_name + '.meta.zip',
            'md5': '76e3d7ed5291b118372e06379cb2b490',
        },
        {
            'url': source_url + base_name + '.audio.1.zip',
            'md5': '186f6273f8f69ed9dbdc18ad65ac234f',
        },
        {
            'url': source_url + base_name + '.audio.2.zip',
            'md5': '7fd6bb63127f5785874a55aba4e77aa5',
        },
        {
            'url': source_url + base_name + '.audio.3.zip',
            'md5': '61396bede29d7c8c89729a01a6f6b2e2',
        },
        {
            'url': source_url + base_name + '.audio.4.zip',
            'md5': '6ddac89717fcf9c92c451868eed77fe1',
        },
        {
            'url': source_url + base_name + '.audio.5.zip',
            'md5': 'af4820756cdf1a7d4bd6037dc034d384',
        },
        {
            'url': source_url + base_name + '.audio.6.zip',
            'md5': 'ebd11ec24411f2a17a64723bd4aa7fff',
        },
        {
            'url': source_url + base_name + '.audio.7.zip',
            'md5': '2be39a76aeed704d5929d020a2909efd',
        },
        {
            'url': source_url + base_name + '.audio.8.zip',
            'md5': '972d8afe0874720fc2f28086e7cb22a9',
        },
    ]
    label_list = [
        'airport', 'shopping_mall', 'metro_station', 'street_pedestrian',
        'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park'
    ]

    meta_base_path = os.path.join(base_name, base_name + '.meta')
    meta = os.path.join(meta_base_path, 'meta.csv')
    meta_info = collections.namedtuple('META_INFO', (
        'filename_audio', 'filename_video', 'scene_label', 'identifier'))
    subset_meta = {
        'train':
        os.path.join(meta_base_path, 'evaluation_setup', 'fold1_train.csv'),
        'dev':
        os.path.join(meta_base_path, 'evaluation_setup', 'fold1_evaluate.csv'),
        'test':
        os.path.join(meta_base_path, 'evaluation_setup', 'fold1_test.csv'),
    }
    subset_meta_info = collections.namedtuple('SUBSET_META_INFO', (
        'filename_audio', 'filename_video', 'scene_label'))
    audio_path = os.path.join(base_name, 'audio')

    def __init__(self, mode: str='train', feat_type: str='raw', **kwargs):
        """
        Ags:
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train or dev).
            feat_type (:obj:`str`, `optional`, defaults to `raw`):
                It identifies the feature type that user wants to extrace of an audio file.
        """
        files, labels = self._get_data(mode)
        super(UrbanAudioVisualScenes, self).__init__(
            files=files, labels=labels, feat_type=feat_type, **kwargs)

    def _get_meta_info(self, subset: str=None,
                       skip_header: bool=True) -> List[collections.namedtuple]:
        if subset is None:
            meta_file = self.meta
            meta_info = self.meta_info
        else:
            assert subset in self.subset_meta, f'Subset must be one in {list(self.subset_meta.keys())}, but got {subset}.'
            meta_file = self.subset_meta[subset]
            meta_info = self.subset_meta_info

        ret = []
        with open(os.path.join(DATA_HOME, meta_file), 'r') as rf:
            lines = rf.readlines()[1:] if skip_header else rf.readlines()
            for line in lines:
                ret.append(meta_info(*line.strip().split('\t')))
        return ret

    def _get_data(self, mode: str) -> Tuple[List[str], List[int]]:
        if not os.path.isdir(os.path.join(DATA_HOME, self.audio_path)) or \
            not os.path.isfile(os.path.join(DATA_HOME, self.meta)):
            download_and_decompress(self.archieves,
                                    os.path.join(DATA_HOME, self.base_name))

        meta_info = self._get_meta_info(subset=mode, skip_header=True)

        files = []
        labels = []
        for sample in meta_info:
            filename, _, label = sample[:3]
            filename = os.path.basename(filename)
            target = self.label_list.index(label)

            files.append(os.path.join(DATA_HOME, self.audio_path, filename))
            labels.append(int(target))

        return files, labels
