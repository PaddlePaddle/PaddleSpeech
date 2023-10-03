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
from collections import OrderedDict

import io
import os
import kaldiio
import numpy as np
import soundfile
import h5py

from .utility import feat_type
from paddlespeech.audio.transform.transformation import Transformation
from paddlespeech.s2t.utils.log import Log
# from paddlespeech.s2t.frontend.augmentor.augmentation import AugmentationPipeline as Transformation

__all__ = ["LoadInputsAndTargets"]

logger = Log(__name__).getlog()


class LoadInputsAndTargets():
    """Create a mini-batch from a list of dicts

    >>> batch = [('utt1',
    ...           dict(input=[dict(feat='some.ark:123',
    ...                            filetype='mat',
    ...                            name='input1',
    ...                            shape=[100, 80])],
    ...                output=[dict(tokenid='1 2 3 4',
    ...                             name='target1',
    ...                             shape=[4, 31])]]))
    >>> l = LoadInputsAndTargets()
    >>> feat, target = l(batch)

    :param: str mode: Specify the task mode, "asr" or "tts"
    :param: str preprocess_conf: The path of a json file for pre-processing
    :param: bool load_input: If False, not to load the input data
    :param: bool load_output: If False, not to load the output data
    :param: bool sort_in_input_length: Sort the mini-batch in descending order
        of the input length
    :param: bool use_speaker_embedding: Used for tts mode only
    :param: bool use_second_target: Used for tts mode only
    :param: dict preprocess_args: Set some optional arguments for preprocessing
    :param: Optional[dict] preprocess_args: Used for tts mode only
    """

    def __init__(
            self,
            mode="asr",
            preprocess_conf=None,
            load_input=True,
            load_output=True,
            sort_in_input_length=True,
            preprocess_args=None,
            keep_all_data_on_mem=False, ):
        self._loaders = {}

        if mode not in ["asr"]:
            raise ValueError("Only asr are allowed: mode={}".format(mode))

        if preprocess_conf:
            self.preprocessing = Transformation(preprocess_conf)
            logger.warning(
                "[Experimental feature] Some preprocessing will be done "
                "for the mini-batch creation using {}".format(
                    self.preprocessing))
        else:
            # If conf doesn't exist, this function don't touch anything.
            self.preprocessing = None

        self.mode = mode
        self.load_output = load_output
        self.load_input = load_input
        self.sort_in_input_length = sort_in_input_length
        if preprocess_args:
            assert isinstance(preprocess_args, dict), type(preprocess_args)
            self.preprocess_args = dict(preprocess_args)
        else:
            self.preprocess_args = {}
        self.keep_all_data_on_mem = keep_all_data_on_mem

    def __call__(self, batch, return_uttid=False):
        """Function to load inputs and targets from list of dicts

        :param List[Tuple[str, dict]] batch: list of dict which is subset of
            loaded data.json
        :param bool return_uttid: return utterance ID information for visualization
        :return: list of input token id sequences [(L_1), (L_2), ..., (L_B)]
        :return: list of input feature sequences
            [(T_1, D), (T_2, D), ..., (T_B, D)]
        :rtype: list of float ndarray
        :return: list of target token id sequences [(L_1), (L_2), ..., (L_B)]
        :rtype: list of int ndarray

        """
        x_feats_dict = OrderedDict()  # OrderedDict[str, List[np.ndarray]]
        y_feats_dict = OrderedDict()  # OrderedDict[str, List[np.ndarray]]
        uttid_list = []  # List[str]

        for uttid, info in batch:
            uttid_list.append(uttid)

            if self.load_input:
                # Note(kamo): This for-loop is for multiple inputs
                for idx, inp in enumerate(info["input"]):
                    # {"input":
                    #  [{"feat": "some/path.h5:F01_050C0101_PED_REAL",
                    #    "filetype": "hdf5",
                    #    "name": "input1", ...}], ...}
                    x = self._get_from_loader(
                        filepath=inp["feat"],
                        filetype=inp.get("filetype", "mat"))
                    x_feats_dict.setdefault(inp["name"], []).append(x)

            if self.load_output:
                for idx, inp in enumerate(info["output"]):
                    if "tokenid" in inp:
                        # ======= Legacy format for output =======
                        # {"output": [{"tokenid": "1 2 3 4"}])
                        x = np.fromiter(
                            map(int, inp["tokenid"].split()), dtype=np.int64)
                    else:
                        # ======= New format =======
                        # {"input":
                        #  [{"feat": "some/path.h5:F01_050C0101_PED_REAL",
                        #    "filetype": "hdf5",
                        #    "name": "target1", ...}], ...}
                        x = self._get_from_loader(
                            filepath=inp["feat"],
                            filetype=inp.get("filetype", "mat"))

                    y_feats_dict.setdefault(inp["name"], []).append(x)

        if self.mode == "asr":
            return_batch, uttid_list = self._create_batch_asr(
                x_feats_dict, y_feats_dict, uttid_list)
        else:
            raise NotImplementedError(self.mode)

        if self.preprocessing is not None:
            # Apply pre-processing all input features
            for x_name in return_batch.keys():
                if x_name.startswith("input"):
                    return_batch[x_name] = self.preprocessing(
                        return_batch[x_name], uttid_list,
                        **self.preprocess_args)

        if return_uttid:
            return tuple(return_batch.values()), uttid_list

        # Doesn't return the names now.
        return tuple(return_batch.values())

    def _create_batch_asr(self, x_feats_dict, y_feats_dict, uttid_list):
        """Create a OrderedDict for the mini-batch

        :param OrderedDict x_feats_dict:
            e.g. {"input1": [ndarray, ndarray, ...],
                  "input2": [ndarray, ndarray, ...]}
        :param OrderedDict y_feats_dict:
            e.g. {"target1": [ndarray, ndarray, ...],
                  "target2": [ndarray, ndarray, ...]}
        :param: List[str] uttid_list:
            Give uttid_list to sort in the same order as the mini-batch
        :return: batch, uttid_list
        :rtype: Tuple[OrderedDict, List[str]]
        """
        # handle single-input and multi-input (paralell) asr mode
        xs = list(x_feats_dict.values())

        if self.load_output:
            ys = list(y_feats_dict.values())
            assert len(xs[0]) == len(ys[0]), (len(xs[0]), len(ys[0]))

            # get index of non-zero length samples
            nonzero_idx = list(
                filter(lambda i: len(ys[0][i]) > 0, range(len(ys[0]))))
            for n in range(1, len(y_feats_dict)):
                nonzero_idx = filter(lambda i: len(ys[n][i]) > 0, nonzero_idx)
        else:
            # Note(kamo): Be careful not to make nonzero_idx to a generator
            nonzero_idx = list(range(len(xs[0])))

        if self.sort_in_input_length:
            # sort in input lengths based on the first input
            nonzero_sorted_idx = sorted(
                nonzero_idx, key=lambda i: -len(xs[0][i]))
        else:
            nonzero_sorted_idx = nonzero_idx

        if len(nonzero_sorted_idx) != len(xs[0]):
            logger.warning(
                "Target sequences include empty tokenid (batch {} -> {}).".
                format(len(xs[0]), len(nonzero_sorted_idx)))

        # remove zero-length samples
        xs = [[x[i] for i in nonzero_sorted_idx] for x in xs]
        uttid_list = [uttid_list[i] for i in nonzero_sorted_idx]

        x_names = list(x_feats_dict.keys())
        if self.load_output:
            ys = [[y[i] for i in nonzero_sorted_idx] for y in ys]
            y_names = list(y_feats_dict.keys())

            # Keeping x_name and y_name, e.g. input1, for future extension
            return_batch = OrderedDict([
                * [(x_name, x) for x_name, x in zip(x_names, xs)],
                * [(y_name, y) for y_name, y in zip(y_names, ys)],
            ])
        else:
            return_batch = OrderedDict(
                [(x_name, x) for x_name, x in zip(x_names, xs)])
        return return_batch, uttid_list

    def _get_from_loader(self, filepath, filetype):
        """Return ndarray

        In order to make the fds to be opened only at the first referring,
        the loader are stored in self._loaders

        >>> ndarray = loader.get_from_loader(
        ...     'some/path.h5:F01_050C0101_PED_REAL', filetype='hdf5')

        :param: str filepath:
        :param: str filetype:
        :return:
        :rtype: np.ndarray
        """
        if filetype == "hdf5":
            # e.g.
            #    {"input": [{"feat": "some/path.h5:F01_050C0101_PED_REAL",
            #                "filetype": "hdf5",
            # -> filepath = "some/path.h5", key = "F01_050C0101_PED_REAL"
            filepath, key = filepath.split(":", 1)

            loader = self._loaders.get(filepath)
            if loader is None:
                # To avoid disk access, create loader only for the first time
                loader = h5py.File(filepath, "r")
                self._loaders[filepath] = loader
            return loader[key][()]
        elif filetype == "sound.hdf5":
            # e.g.
            #    {"input": [{"feat": "some/path.h5:F01_050C0101_PED_REAL",
            #                "filetype": "sound.hdf5",
            # -> filepath = "some/path.h5", key = "F01_050C0101_PED_REAL"
            filepath, key = filepath.split(":", 1)

            loader = self._loaders.get(filepath)
            if loader is None:
                # To avoid disk access, create loader only for the first time
                loader = SoundHDF5File(filepath, "r", dtype="int16")
                self._loaders[filepath] = loader
            array, rate = loader[key]
            return array
        elif filetype == "sound":
            # e.g.
            #    {"input": [{"feat": "some/path.wav",
            #                "filetype": "sound"},
            # Assume PCM16
            if not self.keep_all_data_on_mem:
                array, _ = soundfile.read(filepath, dtype="int16")
                return array
            if filepath not in self._loaders:
                array, _ = soundfile.read(filepath, dtype="int16")
                self._loaders[filepath] = array
            return self._loaders[filepath]
        elif filetype == "npz":
            # e.g.
            #    {"input": [{"feat": "some/path.npz:F01_050C0101_PED_REAL",
            #                "filetype": "npz",
            filepath, key = filepath.split(":", 1)

            loader = self._loaders.get(filepath)
            if loader is None:
                # To avoid disk access, create loader only for the first time
                loader = np.load(filepath)
                self._loaders[filepath] = loader
            return loader[key]
        elif filetype == "npy":
            # e.g.
            #    {"input": [{"feat": "some/path.npy",
            #                "filetype": "npy"},
            if not self.keep_all_data_on_mem:
                return np.load(filepath)
            if filepath not in self._loaders:
                self._loaders[filepath] = np.load(filepath)
            return self._loaders[filepath]
        elif filetype in ["mat", "vec"]:
            # e.g.
            #    {"input": [{"feat": "some/path.ark:123",
            #                "filetype": "mat"}]},
            # In this case, "123" indicates the starting points of the matrix
            # load_mat can load both matrix and vector
            if not self.keep_all_data_on_mem:
                return kaldiio.load_mat(filepath)
            if filepath not in self._loaders:
                self._loaders[filepath] = kaldiio.load_mat(filepath)
            return self._loaders[filepath]
        elif filetype == "scp":
            # e.g.
            #    {"input": [{"feat": "some/path.scp:F01_050C0101_PED_REAL",
            #                "filetype": "scp",
            filepath, key = filepath.split(":", 1)
            loader = self._loaders.get(filepath)
            if loader is None:
                # To avoid disk access, create loader only for the first time
                loader = kaldiio.load_scp(filepath)
                self._loaders[filepath] = loader
            return loader[key]
        else:
            raise NotImplementedError(
                "Not supported: loader_type={}".format(filetype))

    def file_type(self, filepath):
        return feat_type(filepath)


class SoundHDF5File():
    """Collecting sound files to a HDF5 file

    >>> f = SoundHDF5File('a.flac.h5', mode='a')
    >>> array = np.random.randint(0, 100, 100, dtype=np.int16)
    >>> f['id'] = (array, 16000)
    >>> array, rate = f['id']


    :param: str filepath:
    :param: str mode:
    :param: str format: The type used when saving wav. flac, nist, htk, etc.
    :param: str dtype:

    """

    def __init__(self,
                 filepath,
                 mode="r+",
                 format=None,
                 dtype="int16",
                 **kwargs):
        self.filepath = filepath
        self.mode = mode
        self.dtype = dtype

        self.file = h5py.File(filepath, mode, **kwargs)
        if format is None:
            # filepath = a.flac.h5 -> format = flac
            second_ext = os.path.splitext(os.path.splitext(filepath)[0])[1]
            format = second_ext[1:]
            if format.upper() not in soundfile.available_formats():
                # If not found, flac is selected
                format = "flac"

        # This format affects only saving
        self.format = format

    def __repr__(self):
        return '<SoundHDF5 file "{}" (mode {}, format {}, type {})>'.format(
            self.filepath, self.mode, self.format, self.dtype)

    def create_dataset(self, name, shape=None, data=None, **kwds):
        f = io.BytesIO()
        array, rate = data
        soundfile.write(f, array, rate, format=self.format)
        self.file.create_dataset(
            name, shape=shape, data=np.void(f.getvalue()), **kwds)

    def __setitem__(self, name, data):
        self.create_dataset(name, data=data)

    def __getitem__(self, key):
        data = self.file[key][()]
        f = io.BytesIO(data.tobytes())
        array, rate = soundfile.read(f, dtype=self.dtype)
        return array, rate

    def keys(self):
        return self.file.keys()

    def values(self):
        for k in self.file:
            yield self[k]

    def items(self):
        for k in self.file:
            yield k, self[k]

    def __iter__(self):
        return iter(self.file)

    def __contains__(self, item):
        return item in self.file

    def __len__(self):
        return len(self.file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def close(self):
        self.file.close()
