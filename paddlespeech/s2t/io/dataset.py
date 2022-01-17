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
# Modified from espnet(https://github.com/espnet/espnet)
# Modified from wenet(https://github.com/wenet-e2e/wenet)

import jsonlines
from paddle.io import Dataset
import csv
import re
from paddlespeech.s2t.frontend.utility import read_manifest
from paddlespeech.s2t.utils.log import Log

__all__ = [
    "ManifestDataset", 
    "TransformDataset",
    "CSVManifestDataset"]

logger = Log(__name__).getlog()

class CSVManifestDataset(Dataset):
    def __init__(self,
                 data):
        super().__init__()
        self.data = data
        
        # 获取音频的索引id
        self.data_ids = list(self.data.keys())

        # static_keys 中保存了数据集中音频信息字段
        static_keys = list(self.data[self.data_ids[0]].keys())
        logger.info("static_keys: {}".format(static_keys))
        if "id" in static_keys:
            raise ValueError("The key 'id' is reserved for the data point id.")
        else:
            static_keys.append("id")
        logger.info("final static_keys: {}".format(static_keys))
        # self.spk_ids = self.data["spk_id"]
        self.update_utt_labels()
        # logger.info("spk ids: {}".format(self.labels))
    @classmethod
    def from_config(cls, config):
        manifest_path = config.manifest
        data = load_data_csv(manifest_path)
        dataset = cls(data)
        return dataset

    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        utt_id = self.data_ids[idx]
        data_point = self.data[utt_id]
        # return data_point
        data_point["id"] = utt_id
        return data_point
        
        # return self.compute_outputs(data_point)
    def update_utt_labels(self):
        self.labels = []
        for utt in self.data_ids:
            label = self.data[utt]["spk_id"]
            if label not in self.labels:
                self.labels.append(label)
    # def compute_outputs(self, data):
        # 这里是动态计算，导致每次训练的时候特征要重复计算
        # print("compute outputs, data: {}".format(data))
        # duration = data["duration"]
        # start = data["start"]
        # stop = data["stop"]
        # wav_path = data["wav"]
        # logger.info("wav path: {}".format(wav_path))
        # num_samples = stop - start

        # return wav_path

class ManifestDataset(Dataset):
    @classmethod
    def from_config(cls, config):
        """Build a ManifestDataset object from a config.

        Args:
            config (yacs.config.CfgNode): configs object.

        Returns:
            ManifestDataset: dataet object.
        """
        assert 'manifest' in config
        assert config.manifest

        dataset = cls(
            manifest_path=config.manifest,
            max_input_len=config.max_input_len,
            min_input_len=config.min_input_len,
            max_output_len=config.max_output_len,
            min_output_len=config.min_output_len,
            max_output_input_ratio=config.max_output_input_ratio,
            min_output_input_ratio=config.min_output_input_ratio, )
        return dataset

    def __init__(self,
                 manifest_path,
                 max_input_len=float('inf'),
                 min_input_len=0.0,
                 max_output_len=float('inf'),
                 min_output_len=0.0,
                 max_output_input_ratio=float('inf'),
                 min_output_input_ratio=0.0):
        """Manifest Dataset

        Args:
            manifest_path (str): manifest josn file path
            max_input_len ([type], optional): maximum output seq length,
                in seconds for raw wav, in frame numbers for feature data. Defaults to float('inf').
            min_input_len (float, optional): minimum input seq length,
                in seconds for raw wav, in frame numbers for feature data. Defaults to 0.0.
            max_output_len (float, optional): maximum input seq length,
                in modeling units. Defaults to 500.0.
            min_output_len (float, optional): minimum input seq length,
                in modeling units. Defaults to 0.0.
            max_output_input_ratio (float, optional): maximum output seq length/output seq length ratio.
                Defaults to 10.0.
            min_output_input_ratio (float, optional): minimum output seq length/output seq length ratio.
                Defaults to 0.05.

        """
        super().__init__()

        # read manifest
        self._manifest = read_manifest(
            manifest_path=manifest_path,
            max_input_len=max_input_len,
            min_input_len=min_input_len,
            max_output_len=max_output_len,
            min_output_len=min_output_len,
            max_output_input_ratio=max_output_input_ratio,
            min_output_input_ratio=min_output_input_ratio)
        self._manifest.sort(key=lambda x: x["input"][0]["shape"][0])

    def __len__(self):
        return len(self._manifest)

    def __getitem__(self, idx):
        return self._manifest[idx]


class TransformDataset(Dataset):
    """Transform Dataset.

    Args:
        data: list object from make_batchset
        converter: batch function
        reader: read data
    """

    def __init__(self, data, converter, reader):
        """Init function."""
        super().__init__()
        self.data = data
        self.converter = converter
        self.reader = reader

    def __len__(self):
        """Len function."""
        return len(self.data)

    def __getitem__(self, idx):
        """[] operator."""
        return self.converter([self.reader(self.data[idx], return_uttid=True)])


class AudioDataset(Dataset):
    def __init__(self,
                 data_file,
                 max_length=10240,
                 min_length=0,
                 token_max_length=200,
                 token_min_length=1,
                 batch_type='static',
                 batch_size=1,
                 max_frames_in_batch=0,
                 sort=True,
                 raw_wav=True,
                 stride_ms=10):
        """Dataset for loading audio data.
        Attributes::
            data_file: input data file
                Plain text data file, each line contains following 7 fields,
                which is split by '\t':
                    utt:utt1
                    feat:tmp/data/file1.wav or feat:tmp/data/fbank.ark:30
                    feat_shape: 4.95(in seconds) or feat_shape:495,80(495 is in frames)
                    text:i love you
                    token: i <space> l o v e <space> y o u
                    tokenid: int id of this token
                    token_shape: M,N    # M is the number of token, N is vocab size
            max_length: drop utterance which is greater than max_length(10ms), unit 10ms.
            min_length: drop utterance which is less than min_length(10ms), unit 10ms.
            token_max_length: drop utterance which is greater than token_max_length,
                especially when use char unit for english modeling
            token_min_length: drop utterance which is less than token_max_length
            batch_type: static or dynamic, see max_frames_in_batch(dynamic)
            batch_size: number of utterances in a batch,
               it's for static batch size.
            max_frames_in_batch: max feature frames in a batch,
               when batch_type is dynamic, it's for dynamic batch size.
               Then batch_size is ignored, we will keep filling the
               batch until the total frames in batch up to max_frames_in_batch.
            sort: whether to sort all data, so the utterance with the same
               length could be filled in a same batch.
            raw_wav: use raw wave or extracted featute.
                if raw wave is used, dynamic waveform-level augmentation could be used
                and the feature is extracted by torchaudio.
                if extracted featute(e.g. by kaldi) is used, only feature-level
                augmentation such as specaug could be used.
        """
        assert batch_type in ['static', 'dynamic']
        # read manifest
        with jsonlines.open(data_file, 'r') as reader:
            data = list(reader)
        if sort:
            data = sorted(data, key=lambda x: x["feat_shape"][0])
        if raw_wav:
            path_suffix = data[0]['feat'].split(':')[0].splitext()[-1]
            assert path_suffix not in ('.ark', '.scp')
            # m second to n frame
            data = list(
                map(lambda x: (float(x['feat_shape'][0]) * 1000 / stride_ms),
                    data))

        self.input_dim = data[0]['feat_shape'][1]
        self.output_dim = data[0]['token_shape'][1]

        valid_data = []
        for i in range(len(data)):
            length = data[i]['feat_shape'][0]
            token_length = data[i]['token_shape'][0]
            # remove too lang or too short utt for both input and output
            # to prevent from out of memory
            if length > max_length or length < min_length:
                pass
            elif token_length > token_max_length or token_length < token_min_length:
                pass
            else:
                valid_data.append(data[i])
        logger.info(f"raw dataset len: {len(data)}")
        data = valid_data
        num_data = len(data)
        logger.info(f"dataset len after filter: {num_data}")

        self.minibatch = []
        # Dynamic batch size
        if batch_type == 'dynamic':
            assert (max_frames_in_batch > 0)
            self.minibatch.append([])
            num_frames_in_batch = 0
            for i in range(num_data):
                length = data[i]['feat_shape'][0]
                num_frames_in_batch += length
                if num_frames_in_batch > max_frames_in_batch:
                    self.minibatch.append([])
                    num_frames_in_batch = length
                self.minibatch[-1].append(data[i])
        # Static batch size
        else:
            cur = 0
            while cur < num_data:
                end = min(cur + batch_size, num_data)
                item = []
                for i in range(cur, end):
                    item.append(data[i])
                self.minibatch.append(item)
                cur = end

    def __len__(self):
        """number of example(batch)"""
        return len(self.minibatch)

    def __getitem__(self, idx):
        """batch example of idx"""
        return self.minibatch[idx]

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


