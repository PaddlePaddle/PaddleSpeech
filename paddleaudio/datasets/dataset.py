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
from typing import List

import numpy as np
import paddle
import csv
import json
import re
from paddle.io import Dataset
from ..backends import load as load_audio
from ..features import melspectrogram
from ..features import mfcc
from paddlespeech.s2t.utils.log import Log 
from paddleaudio.features.audiopipeline import AudioPipeline
logger = Log(__name__).getlog()

feat_funcs = {
    'raw': None,
    'melspectrogram': melspectrogram,
    'mfcc': mfcc,
}


class AudioClassificationDataset(paddle.io.Dataset):
    """
    Base class of audio classification dataset.
    """

    def __init__(self,
                 files: List[str],
                 labels: List[int],
                 feat_type: str='raw',
                 sample_rate: int=None,
                 **kwargs):
        """
        Ags:
            files (:obj:`List[str]`): A list of absolute path of audio files.
            labels (:obj:`List[int]`): Labels of audio files.
            feat_type (:obj:`str`, `optional`, defaults to `raw`):
                It identifies the feature type that user wants to extrace of an audio file.
        """
        super(AudioClassificationDataset, self).__init__()

        if feat_type not in feat_funcs.keys():
            raise RuntimeError(
                f"Unknown feat_type: {feat_type}, it must be one in {list(feat_funcs.keys())}"
            )

        self.files = files
        self.labels = labels

        self.feat_type = feat_type
        self.sample_rate = sample_rate
        self.feat_config = kwargs  # Pass keyword arguments to customize feature config

    def _get_data(self, input_file: str):
        raise NotImplementedError

    def _convert_to_record(self, idx):
        file, label = self.files[idx], self.labels[idx]

        if self.sample_rate is None:
            waveform, sample_rate = load_audio(file)
        else:
            waveform, sample_rate = load_audio(file, sr=self.sample_rate)

        feat_func = feat_funcs[self.feat_type]

        record = {}
        record['feat'] = feat_func(
            waveform, sample_rate,
            **self.feat_config) if feat_func else waveform
        record['label'] = label
        return record

    def __getitem__(self, idx):
        record = self._convert_to_record(idx)
        return np.array(record['feat']).transpose(), np.array(
            record['label'], dtype=np.int64)

    def __len__(self):
        return len(self.files)

class SpeechDataset(Dataset):
    def __init__(self,
                 data,
                 func=None):
        """
        创建音频数据集，这里的数据集是已经经过特征处理之后的数据，模型直接获取相关的特征数据

        Args:
            data ([type]): 每个音频特征数据，是一个字典形式。字典的key是所有音频的id，即utt-id，字典的value是音频的特征数据
        """
        super(SpeechDataset, self).__init__()

        # 获取音频的索引id,即utt-id
        self.data = data
        self.data_ids = list(data.keys())
        self.func = func
        
        # 处理数据得到特征
        if self.func:
            self.data = self.func(data)
        self.update_utt_labels()
        
    @classmethod
    def from_config(cls, config):
        manifest_path = config.manifest
        # 加载json数据
        data = load_from_json(manifest_path)
        # audio_pipeline 对数据进行处理
        audio_pipeline = AudioPipeline(config)
        dataset = cls(data, func=audio_pipeline)
        return dataset

    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        utt_id = self.data_ids[idx]
        data_point = self.data[utt_id]["wav"]
        spk_id = self.data[utt_id]["spk_id"]
        spk_label = self.spk2idx[spk_id]
        # logger.info("data point: {}".format(data_point))
        return data_point, spk_label

    def update_utt_labels(self):
        self.spk2idx = {}
        self.idx2spk = {}
        for utt in self.data_ids:
            spk_id = self.data[utt]["spk_id"]
            idx = len(self.spk2idx)
            self.spk2idx[spk_id] = idx
            self.idx2spk[idx] = spk_id

def load_data_csv(csv_path, replacements={}):
    """
    reference: https://github.com/speechbrain/speechbrain/blob/d3d267e86c3b5494cd970319a63d5dae8c0662d7/speechbrain/dataio/dataio.py#L89
    """
    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        for row in reader:
            # ID:
            try:
                data_id = row["ID"]
                del row["ID"]  # This is used as a key in result, instead.
            except KeyError:
                raise KeyError(
                    "CSV has to have an 'ID' field, with unique ids"
                    " for all data points"
                )
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            for key, value in row.items():
                try:
                    row[key] = variable_finder.sub(
                        lambda match: str(replacements[match[1]]), value
                    )
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements "
                        "which were not supplied."
                    )
            # Duration:
            if "duration" in row:
                row["duration"] = float(row["duration"])
            result[data_id] = row
            # logger.info("data_id: {}, row: {}".format(data_id, row))
    return result

def load_from_json(dataset_path):
    """
    加载 json 格式的数据集，必须包含如下7个字段
    utt-id,duration,wav,start,end,text
    Args:
        dataset_path ([type]): [description]
    """
    try:
        with open(dataset_path, 'r') as f:
            dataset_json = json.load(f)
    except:
        raise ValueError("input an invalid json dataset file: {}".format(dataset_path))

    return dataset_json
