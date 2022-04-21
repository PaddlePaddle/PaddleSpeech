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
import json
from dataclasses import dataclass
from dataclasses import fields

from paddle.io import Dataset

from paddleaudio import load as load_audio
from paddleaudio.compliance.librosa import melspectrogram
from paddleaudio.compliance.librosa import mfcc


@dataclass
class meta_info:
    """the audio meta info in the vector JSONDataset
    Args:
        utt_id (str): the segment name
        duration (float): segment time
        wav (str): wav file path
        start (int): start point in the original wav file
        stop (int): stop point in the original wav file
        lab_id (str): the record id
    """
    utt_id: str
    duration: float
    wav: str
    start: int
    stop: int
    record_id: str


# json dataset support feature type
feat_funcs = {
    'raw': None,
    'melspectrogram': melspectrogram,
    'mfcc': mfcc,
}


class JSONDataset(Dataset):
    """
    dataset from json file.
    """

    def __init__(self, json_file: str, feat_type: str='raw', **kwargs):
        """
        Ags:
            json_file (:obj:`str`): Data prep JSON file.
            labels (:obj:`List[int]`): Labels of audio files.
            feat_type (:obj:`str`, `optional`, defaults to `raw`):
                It identifies the feature type that user wants to extrace of an audio file.
        """
        if feat_type not in feat_funcs.keys():
            raise RuntimeError(
                f"Unknown feat_type: {feat_type}, it must be one in {list(feat_funcs.keys())}"
            )

        self.json_file = json_file
        self.feat_type = feat_type
        self.feat_config = kwargs
        self._data = self._get_data()
        super(JSONDataset, self).__init__()

    def _get_data(self):
        with open(self.json_file, "r") as f:
            meta_data = json.load(f)
        data = []
        for key in meta_data:
            sub_seg = meta_data[key]["wav"]
            wav = sub_seg["file"]
            duration = sub_seg["duration"]
            start = sub_seg["start"]
            stop = sub_seg["stop"]
            rec_id = str(key).rsplit("_", 2)[0]
            data.append(
                meta_info(
                    str(key),
                    float(duration), wav, int(start), int(stop), str(rec_id)))
        return data

    def _convert_to_record(self, idx: int):
        sample = self._data[idx]

        record = {}
        # To show all fields in a namedtuple
        for field in fields(sample):
            record[field.name] = getattr(sample, field.name)

        waveform, sr = load_audio(record['wav'])
        waveform = waveform[record['start']:record['stop']]

        feat_func = feat_funcs[self.feat_type]
        feat = feat_func(
            waveform, sr=sr, **self.feat_config) if feat_func else waveform

        record.update({'feat': feat})

        return record

    def __getitem__(self, idx):
        return self._convert_to_record(idx)

    def __len__(self):
        return len(self._data)
