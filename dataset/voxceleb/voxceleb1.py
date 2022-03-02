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
import csv
import glob
import os
import random
from typing import Dict, List, Tuple

from paddle.io import Dataset
from tqdm import tqdm
from pathos.multiprocessing import Pool

from paddleaudio.backends import load as load_audio
from paddleaudio.utils import DATA_HOME, decompress, download_and_decompress
from paddleaudio.datasets.dataset import feat_funcs
from utils.utility import unpack
from utils.utility import download

__all__ = ['VoxCeleb1']


class VoxCeleb1(Dataset):
    source_url = 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/'
    archieves_audio_dev = [
        {
            'url': source_url + 'vox1_dev_wav_partaa',
            'md5': 'e395d020928bc15670b570a21695ed96',
        },
        {
            'url': source_url + 'vox1_dev_wav_partab',
            'md5': 'bbfaaccefab65d82b21903e81a8a8020',
        },
        {
            'url': source_url + 'vox1_dev_wav_partac',
            'md5': '017d579a2a96a077f40042ec33e51512',
        },
        {
            'url': source_url + 'vox1_dev_wav_partad',
            'md5': '7bb1e9f70fddc7a678fa998ea8b3ba19',
        },
    ]
    archieves_audio_test = [
        {
            'url': source_url + 'vox1_test_wav.zip',
            'md5': '185fdc63c3c739954633d50379a3d102',
        },
    ]
    archieves_meta = [
        {
            'url': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt',
            'md5': 'b73110731c9223c1461fe49cb48dddfc',
        },
    ]


    num_speakers = 1211  # 1211 vox1, 5994 vox2, 7205 vox1+2, test speakers: 41
    sample_rate = 16000
    meta_info = collections.namedtuple(
        'META_INFO', ('id', 'duration', 'wav', 'start', 'stop', 'spk_id'))
    base_path = os.path.join(DATA_HOME, 'vox1')
    wav_path = os.path.join(base_path, 'wav')
    subsets = ['train', 'dev', 'enrol', 'test']

    def __init__(self,
                 subset: str = 'train',
                 feat_type: str = 'raw',
                 random_chunk: bool = True,
                 chunk_duration: float = 3.0,       # seconds
                 split_ratio: float = 0.9,          # train split ratio
                 seed: int = 0,
                 target_dir: str = None,
                 **kwargs):

        assert subset in self.subsets, \
            'Dataset subset must be one in {}, but got {}'.format(self.subsets, subset)

        self.subset = subset
        self.spk_id2label = {}
        self.feat_type = feat_type
        self.feat_config = kwargs
        self.random_chunk = random_chunk
        self.chunk_duration = chunk_duration
        self.split_ratio = split_ratio
        self.target_dir = target_dir if target_dir else self.base_path
        self.csv_path = os.path.join(target_dir, 'csv') if target_dir else os.path.join(self.base_path, 'csv')
        self.meta_path = os.path.join(target_dir, 'meta') if target_dir else os.path.join(base_path, 'meta')
        self.veri_test_file = os.path.join(self.meta_path, 'veri_test2.txt')
        # self._data = self._get_data()[:1000]  # KP: Small dataset test.
        self._data = self._get_data()
        super(VoxCeleb1, self).__init__()

        # Set up a seed to reproduce training or predicting result.
        # random.seed(seed)

    def _get_data(self):
        # Download audio files.
        # We need the users to decompress all vox1/dev/wav and vox1/test/wav/ to vox1/wav/ dir
        # so, we check the vox1/wav dir status
        print("wav base path: {}".format(self.wav_path))
        if not os.path.isdir(self.wav_path):
            print("start to download the voxceleb1 dataset")
            download_and_decompress(    # multi-zip parts concatenate to vox1_dev_wav.zip
                self.archieves_audio_dev, self.base_path, decompress=False)
            download_and_decompress(    # download the vox1_test_wav.zip and unzip
                self.archieves_audio_test, self.base_path, decompress=True)

            # Download all parts and concatenate the files into one zip file.
            dev_zipfile = os.path.join(self.base_path, 'vox1_dev_wav.zip')
            print(f'Concatenating all parts to: {dev_zipfile}')
            os.system(
                f'cat {os.path.join(self.base_path, "vox1_dev_wav_parta*")} > {dev_zipfile}'
            )

            # Extract all audio files of dev and test set.
            decompress(dev_zipfile, self.base_path)

        # Download meta files.
        if not os.path.isdir(self.meta_path):
            download_and_decompress(
                self.archieves_meta, self.meta_path, decompress=False)
        
        # Data preparation.
        if not os.path.isdir(self.csv_path):
            os.makedirs(self.csv_path)
            self.prepare_data()

        data = []
        with open(os.path.join(self.csv_path, f'{self.subset}.csv'), 'r') as rf:
            for line in rf.readlines()[1:]:
                audio_id, duration, wav, start, stop, spk_id = line.strip(
                ).split(',')
                data.append(
                    self.meta_info(audio_id, float(duration), wav, int(start),
                                   int(stop), spk_id))

        with open(os.path.join(self.meta_path, 'spk_id2label.txt'), 'r') as f:
            for line in f.readlines():
                spk_id, label = line.strip().split(' ')
                self.spk_id2label[spk_id] = int(label)

        return data

    def _convert_to_record(self, idx: int):
        sample = self._data[idx]

        record = {}
        # To show all fields in a namedtuple: `type(sample)._fields`
        for field in type(sample)._fields:
            record[field] = getattr(sample, field)

        waveform, sr = load_audio(record['wav'])

        # random select a chunk audio samples from the audio
        if self.random_chunk:
            num_wav_samples = waveform.shape[0]
            num_chunk_samples = int(self.chunk_duration * sr)
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
        else:
            start = record['start']
            stop = record['stop']

        waveform = waveform[start:stop]

        assert self.feat_type in feat_funcs.keys(), \
            f"Unknown feat_type: {self.feat_type}, it must be one in {list(feat_funcs.keys())}"
        feat_func = feat_funcs[self.feat_type]
        feat = feat_func(
            waveform, sr=sr, **self.feat_config) if feat_func else waveform

        record.update({'feat': feat})
        if self.subset in ['train',
                           'dev']:  # Labels are available in train and dev.
            record.update({'label': self.spk_id2label[record['spk_id']]})

        return record

    @staticmethod
    def _get_chunks(seg_dur, audio_id, audio_duration):
        num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

        chunk_lst = [
            audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
            for i in range(num_chunks)
        ]
        return chunk_lst

    def _get_audio_info(self, wav_file: str,
                        split_chunks: bool) -> List[List[str]]:
        waveform, sr = load_audio(wav_file)
        spk_id, sess_id, utt_id = wav_file.split("/")[-3:]
        audio_id = '-'.join([spk_id, sess_id, utt_id.split(".")[0]])
        audio_duration = waveform.shape[0] / sr

        ret = []
        if split_chunks:  # Split into pieces of self.chunk_duration seconds.
            uniq_chunks_list = self._get_chunks(self.chunk_duration, audio_id,
                                                audio_duration)

            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]  # Timestamps of start and end
                start_sample = int(float(s) * sr)
                end_sample = int(float(e) * sr)
                # id, duration, wav, start, stop, spk_id
                ret.append([
                    chunk, audio_duration, wav_file, start_sample, end_sample,
                    spk_id
                ])
        else:  # Keep whole audio.
            ret.append([
                audio_id, audio_duration, wav_file, 0, waveform.shape[0], spk_id
            ])
        return ret

    def generate_csv(self,
                     wav_files: List[str],
                     output_file: str,
                     split_chunks: bool = True):
        print(f'Generating csv: {output_file}')
        header = ["id", "duration", "wav", "start", "stop", "spk_id"]

        with Pool(64) as p:
            infos = list(
                tqdm(
                    p.imap(lambda x: self._get_audio_info(x, split_chunks), wav_files), total=len(wav_files)))

        csv_lines = []
        for info in infos:
            csv_lines.extend(info)

        with open(output_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(header)
            for line in csv_lines:
                csv_writer.writerow(line)

    def prepare_data(self):
        # Audio of speakers in veri_test_file should not be included in training set.
        print("start to prepare the data csv file")
        enrol_files = set()
        test_files = set()
        # get the enroll and test audio file path
        with open(self.veri_test_file, 'r') as f:
            for line in f.readlines():
                _, enrol_file, test_file = line.strip().split(' ')
                enrol_files.add(os.path.join(self.wav_path, enrol_file))
                test_files.add(os.path.join(self.wav_path, test_file))
            enrol_files = sorted(enrol_files)
            test_files = sorted(test_files)

        # get the enroll and test speakers
        test_spks = set()
        for file in (enrol_files + test_files):
            spk = file.split('/wav/')[1].split('/')[0]
            test_spks.add(spk)

        # get all the train and dev audios file path
        audio_files = []
        speakers = set()
        for path in [self.wav_path]:
            for file in glob.glob(os.path.join(path, "**", "*.wav"), recursive=True):
                spk = file.split('/wav/')[1].split('/')[0]
                if spk in test_spks:
                    continue
                speakers.add(spk)
                audio_files.append(file)

        print("start to generate the {}".format(os.path.join(self.meta_path, 'spk_id2label.txt')))
        # encode the train and dev speakers label to spk_id2label.txt
        with open(os.path.join(self.meta_path, 'spk_id2label.txt'), 'w') as f:
            for label, spk_id in enumerate(sorted(speakers)):  # 1211 vox1, 5994 vox2, 7205 vox1+2
                f.write(f'{spk_id} {label}\n')

        audio_files = sorted(audio_files)
        random.shuffle(audio_files)
        split_idx = int(self.split_ratio * len(audio_files))
        # split_ratio to train
        train_files, dev_files = audio_files[:split_idx], audio_files[split_idx:]

        self.generate_csv(train_files, 
                          os.path.join(self.csv_path, 'train.csv'))
        self.generate_csv(dev_files, 
                          os.path.join(self.csv_path, 'dev.csv'))
        self.generate_csv(enrol_files,
                          os.path.join(self.csv_path, 'enrol.csv'),
                          split_chunks=False)
        self.generate_csv(test_files,
                          os.path.join(self.csv_path, 'test.csv'),
                          split_chunks=False)

    def __getitem__(self, idx):
        return self._convert_to_record(idx)

    def __len__(self):
        return len(self._data)
