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
from paddle.io import DataLoader

from deepspeech.frontend.utility import read_manifest
from deepspeech.io.batchfy import make_batchset
from deepspeech.io.dataset import TransformDataset
from deepspeech.io.reader import CustomConverter
from deepspeech.io.reader import LoadInputsAndTargets
from deepspeech.utils.log import Log

__all__ = ["BatchDataLoader"]

logger = Log(__name__).getlog()


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
        data_json = read_manifest(json_file)
        logger.info(f"load {json_file} file.")

        # make minibatch list (variable length)
        self.data = make_batchset(
            data_json,
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
        logger.info(f"batchfy data {json_file}: {len(self.data)}.")

        self.load = LoadInputsAndTargets(
            mode="asr",
            load_output=True,
            preprocess_conf=preprocess_conf,
            preprocess_args={"train":
                             train_mode},  # Switch the mode of preprocessing
        )

        # Setup a converter
        if num_encs == 1:
            self.converter = CustomConverter(
                subsampling_factor=subsampling_factor, dtype=dtype)
        else:
            assert NotImplementedError("not impl CustomConverterMulEnc.")

        # hack to make batchsize argument as 1
        # actual bathsize is included in a list
        # default collate function converts numpy array to pytorch tensor
        # we used an empty collate function instead which returns list
        self.train_loader = DataLoader(
            dataset=TransformDataset(
                self.data, lambda data: self.converter([self.load(data, return_uttid=True)])),
            batch_size=1,
            shuffle=not use_sortagrad if train_mode else False,
            collate_fn=lambda x: x[0],
            num_workers=n_iter_processes, )
        logger.info(f"dataloader for {json_file}.")

    def __repr__(self):
        return f"DataLoader {self.json_file}-{self.train_mode}-{self.use_sortagrad}"
