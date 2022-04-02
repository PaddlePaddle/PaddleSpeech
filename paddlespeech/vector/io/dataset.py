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
from dataclasses import dataclass
from dataclasses import fields
from paddle.io import Dataset

from paddleaudio import load as load_audio
from paddlespeech.s2t.utils.log import Log
logger = Log(__name__).getlog()

# the audio meta info in the vector CSVDataset
# utt_id: the utterance segment name
# duration: utterance segment time
# wav: utterance file path
# start: start point in the original wav file
# stop: stop point in the original wav file
# lab_id: the utterance segment's label id


@dataclass
class meta_info:
    """the audio meta info in the vector CSVDataset

    Args:
        utt_id (str): the utterance segment name
        duration (float): utterance segment time
        wav (str): utterance file path
        start (int): start point in the original wav file
        stop (int): stop point in the original wav file
        lab_id (str): the utterance segment's label id
    """
    utt_id: str
    duration: float
    wav: str
    start: int
    stop: int
    lab_id: str


class CSVDataset(Dataset):
    def __init__(self, csv_path, spk_id2label_path=None, config=None):
        """Implement the CSV Dataset

        Args:
            csv_path (str): csv dataset file path
            spk_id2label_path (str): the utterance label to integer id map file path
            config (CfgNode): yaml config
        """
        super().__init__()
        self.csv_path = csv_path
        self.spk_id2label_path = spk_id2label_path
        self.config = config
        self.spk_id2label = {}
        self.label2spk_id = {}
        self.data = self.load_data_csv()
        self.load_speaker_to_label()

    def load_data_csv(self):
        """Load the csv dataset content and store them in the data property
        the csv dataset's format has six fields, 
        that is audio_id or utt_id, audio duration, segment start point, segment stop point 
        and utterance label.
        Note in training period, the utterance label must has a map to integer id in spk_id2label_path 
        """
        data = []

        with open(self.csv_path, 'r') as rf:
            for line in rf.readlines()[1:]:
                audio_id, duration, wav, start, stop, spk_id = line.strip(
                ).split(',')
                data.append(
                    meta_info(audio_id,
                              float(duration), wav,
                              int(start), int(stop), spk_id))
        return data

    def load_speaker_to_label(self):
        """Load the utterance label map content.
        In vector domain, we call the utterance label as speaker label.
        The speaker label is real speaker label in speaker verification domain,
        and in language identification is language label.
        """
        if not self.spk_id2label_path:
            logger.warning("No speaker id to label file")
            return
        self.spk_id2label = {}
        self.label2spk_id = {}
        with open(self.spk_id2label_path, 'r') as f:
            for line in f.readlines():
                spk_id, label = line.strip().split(' ')
                self.spk_id2label[spk_id] = int(label)
                self.label2spk_id[int(label)] = spk_id

    def convert_to_record(self, idx: int):
        """convert the dataset sample to training record the CSV Dataset

        Args:
            idx (int) : the request index in all the dataset
        """
        sample = self.data[idx]

        record = {}
        # To show all fields in a namedtuple: `type(sample)._fields`
        for field in fields(sample):
            record[field.name] = getattr(sample, field.name)

        waveform, sr = load_audio(record['wav'])

        # random select a chunk audio samples from the audio
        if self.config and self.config.random_chunk:
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
        if self.spk_id2label:
            record.update({'label': self.spk_id2label[record['lab_id']]})

        return record

    def __getitem__(self, idx):
        """Return the specific index sample

        Args:
            idx (int) : the request index in all the dataset
        """
        return self.convert_to_record(idx)

    def __len__(self):
        """Return the dataset length
        """
        return len(self.data)
