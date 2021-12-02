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
from typing import Any
from typing import Dict
from typing import List
from typing import Text

import jsonlines
import numpy as np
from paddle.io import DataLoader

from paddlespeech.s2t.io.batchfy import make_batchset
from paddlespeech.s2t.io.converter import CustomConverter
from paddlespeech.s2t.io.dataset import TransformDataset
from paddlespeech.s2t.io.reader import LoadInputsAndTargets
from paddlespeech.s2t.utils.log import Log

__all__ = ["BatchDataLoader"]

logger = Log(__name__).getlog()


def feat_dim_and_vocab_size(data_json: List[Dict[Text, Any]],
                            mode: Text="asr",
                            iaxis=0,
                            oaxis=0):
    if mode == 'asr':
        feat_dim = data_json[0]['input'][oaxis]['shape'][1]
        vocab_size = data_json[0]['output'][oaxis]['shape'][1]
    else:
        raise ValueError(f"{mode} mode not support!")
    return feat_dim, vocab_size


def batch_collate(x):
    """de-minibatch, since user compose batch.

    Args:
        x (List[Tuple]): [(utts, xs, ilens, ys, olens)]

    Returns:
        Tuple: (utts, xs, ilens, ys, olens)
    """
    return x[0]


class BatchDataLoader():
    def __init__(self,
                 json_file: str,
                 train_mode: bool,
                 sortagrad: bool=False,
                 batch_size: int=0,
                 maxlen_in: float=float('inf'),
                 maxlen_out: float=float('inf'),
                 minibatches: int=0,
                 mini_batch_size: int=1,
                 batch_count: str='auto',
                 batch_bins: int=0,
                 batch_frames_in: int=0,
                 batch_frames_out: int=0,
                 batch_frames_inout: int=0,
                 preprocess_conf=None,
                 n_iter_processes: int=1,
                 subsampling_factor: int=1,
                 num_encs: int=1):
        self.json_file = json_file
        self.train_mode = train_mode
        self.use_sortagrad = sortagrad == -1 or sortagrad > 0
        self.batch_size = batch_size
        self.maxlen_in = maxlen_in
        self.maxlen_out = maxlen_out
        self.batch_count = batch_count
        self.batch_bins = batch_bins
        self.batch_frames_in = batch_frames_in
        self.batch_frames_out = batch_frames_out
        self.batch_frames_inout = batch_frames_inout
        self.subsampling_factor = subsampling_factor
        self.num_encs = num_encs
        self.preprocess_conf = preprocess_conf
        self.n_iter_processes = n_iter_processes

        # read json data
        with jsonlines.open(json_file, 'r') as reader:
            self.data_json = list(reader)

        self.feat_dim, self.vocab_size = feat_dim_and_vocab_size(
            self.data_json, mode='asr')

        # make minibatch list (variable length)
        self.minibaches = make_batchset(
            self.data_json,
            batch_size,
            maxlen_in,
            maxlen_out,
            minibatches,  # for debug
            min_batch_size=mini_batch_size,
            shortest_first=self.use_sortagrad,
            count=batch_count,
            batch_bins=batch_bins,
            batch_frames_in=batch_frames_in,
            batch_frames_out=batch_frames_out,
            batch_frames_inout=batch_frames_inout,
            iaxis=0,
            oaxis=0, )

        # data reader
        self.reader = LoadInputsAndTargets(
            mode="asr",
            load_output=True,
            preprocess_conf=preprocess_conf,
            preprocess_args={"train":
                             train_mode},  # Switch the mode of preprocessing
        )

        # Setup a converter
        if num_encs == 1:
            self.converter = CustomConverter(
                subsampling_factor=subsampling_factor, dtype=np.float32)
        else:
            assert NotImplementedError("not impl CustomConverterMulEnc.")

        # hack to make batchsize argument as 1
        # actual bathsize is included in a list
        # default collate function converts numpy array to pytorch tensor
        # we used an empty collate function instead which returns list
        self.dataset = TransformDataset(self.minibaches, self.converter,
                                        self.reader)

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=1,
            shuffle=not self.use_sortagrad if self.train_mode else False,
            collate_fn=batch_collate,
            num_workers=self.n_iter_processes, )

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.dataloader.__iter__()

    def __call__(self):
        return self.__iter__()

    def __repr__(self):
        echo = f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}> "
        echo += f"train_mode: {self.train_mode}, "
        echo += f"sortagrad: {self.use_sortagrad}, "
        echo += f"batch_size: {self.batch_size}, "
        echo += f"maxlen_in: {self.maxlen_in}, "
        echo += f"maxlen_out: {self.maxlen_out}, "
        echo += f"batch_count: {self.batch_count}, "
        echo += f"batch_bins: {self.batch_bins}, "
        echo += f"batch_frames_in: {self.batch_frames_in}, "
        echo += f"batch_frames_out: {self.batch_frames_out}, "
        echo += f"batch_frames_inout: {self.batch_frames_inout}, "
        echo += f"subsampling_factor: {self.subsampling_factor}, "
        echo += f"num_encs: {self.num_encs}, "
        echo += f"num_workers: {self.n_iter_processes}, "
        echo += f"file: {self.json_file}"
        return echo
