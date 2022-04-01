# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.io import Dataset

from paddleaudio import load as load_audio


class VoxCelebDataset(Dataset):
    meta_info = collections.namedtuple(
        'META_INFO', ('id', 'duration', 'wav', 'start', 'stop', 'spk_id'))

    def __init__(self, csv_path, spk_id2label_path, config):
        super().__init__()
        self.csv_path = csv_path
        self.spk_id2label_path = spk_id2label_path
        self.config = config
        self.data = self.load_data_csv()
        self.spk_id2label = self.load_speaker_to_label()

    def load_data_csv(self):
        data = []
        with open(self.csv_path, 'r') as rf:
            for line in rf.readlines()[1:]:
                audio_id, duration, wav, start, stop, spk_id = line.strip(
                ).split(',')
                data.append(
                    self.meta_info(audio_id,
                                   float(duration), wav,
                                   int(start), int(stop), spk_id))
        return data

    def load_speaker_to_label(self):
        with open(self.spk_id2label_path, 'r') as f:
            for line in f.readlines():
                spk_id, label = line.strip().split(' ')
                self.spk_id2label[spk_id] = int(label)

    def convert_to_record(self, idx: int):
        sample = self.data[idx]

        record = {}
        # To show all fields in a namedtuple: `type(sample)._fields`
        for field in type(sample)._fields:
            record[field] = getattr(sample, field)

        waveform, sr = load_audio(record['wav'])

        # random select a chunk audio samples from the audio
        if self.config.random_chunk:
            num_wav_samples = waveform.shape[0]
            num_chunk_samples = int(self.config.chunk_duration * sr)
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
        else:
            start = record['start']
            stop = record['stop']

        # we only return the waveform as feat
        waveform = waveform[start:stop]
        record.update({'feat': waveform})
        record.update({'label': self.spk_id2label[record['spk_id']]})

        return record

    def __getitem__(self, idx):
        return self.convert_to_record(idx)

    def __len__(self):
        return len(self.data)


class RIRSNoiseDataset(Dataset):
    meta_info = collections.namedtuple('META_INFO', ('id', 'duration', 'wav'))

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.data = self.load_csv_data()

    def load_csv_data(self):
        data = []
        with open(self.csv_path, 'r') as rf:
            for line in rf.readlines()[1:]:
                audio_id, duration, wav = line.strip().split(',')
                data.append(self.meta_info(audio_id, float(duration), wav))

        random.shuffle(data)
        return data

    def convert_to_record(self, idx: int):
        sample = self.data[idx]

        record = {}
        # To show all fields in a namedtuple: `type(sample)._fields`
        for field in type(sample)._fields:
            record[field] = getattr(sample, field)

        waveform, sr = load_audio(record['wav'])

        record.update({'feat': waveform})
        return record

    def __getitem__(self, idx):
        return self.convert_to_record(idx)

    def __len__(self):
        return len(self.data)
