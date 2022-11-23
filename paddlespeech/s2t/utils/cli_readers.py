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
import io
import logging
import sys

import h5py
import kaldiio
import soundfile

from paddlespeech.s2t.io.reader import SoundHDF5File


def file_reader_helper(
        rspecifier: str,
        filetype: str="mat",
        return_shape: bool=False,
        segments: str=None, ):
    """Read uttid and array in kaldi style

    This function might be a bit confusing as "ark" is used
    for HDF5 to imitate "kaldi-rspecifier".

    Args:
        rspecifier: Give as "ark:feats.ark" or "scp:feats.scp"
        filetype: "mat" is kaldi-martix, "hdf5": HDF5
        return_shape: Return the shape of the matrix,
            instead of the matrix. This can reduce IO cost for HDF5.
        segments (str): The file format is
            "<segment-id> <recording-id> <start-time> <end-time>\n"
            "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5\n"
    Returns:
        Generator[Tuple[str, np.ndarray], None, None]:

    Examples:
        Read from kaldi-matrix ark file:

        >>> for u, array in file_reader_helper('ark:feats.ark', 'mat'):
        ...     array

        Read from HDF5 file:

        >>> for u, array in file_reader_helper('ark:feats.h5', 'hdf5'):
        ...     array

    """
    if filetype == "mat":
        return KaldiReader(
            rspecifier, return_shape=return_shape, segments=segments)
    elif filetype == "hdf5":
        return HDF5Reader(rspecifier, return_shape=return_shape)
    elif filetype == "sound.hdf5":
        return SoundHDF5Reader(rspecifier, return_shape=return_shape)
    elif filetype == "sound":
        return SoundReader(rspecifier, return_shape=return_shape)
    else:
        raise NotImplementedError(f"filetype={filetype}")


class KaldiReader:
    def __init__(self, rspecifier, return_shape=False, segments=None):
        self.rspecifier = rspecifier
        self.return_shape = return_shape
        self.segments = segments

    def __iter__(self):
        with kaldiio.ReadHelper(
                self.rspecifier, segments=self.segments) as reader:
            for key, array in reader:
                if self.return_shape:
                    array = array.shape
                yield key, array


class HDF5Reader:
    def __init__(self, rspecifier, return_shape=False):
        if ":" not in rspecifier:
            raise ValueError('Give "rspecifier" such as "ark:some.ark: {}"'.
                             format(self.rspecifier))
        self.rspecifier = rspecifier
        self.ark_or_scp, self.filepath = self.rspecifier.split(":", 1)
        if self.ark_or_scp not in ["ark", "scp"]:
            raise ValueError(f"Must be scp or ark: {self.ark_or_scp}")

        self.return_shape = return_shape

    def __iter__(self):
        if self.ark_or_scp == "scp":
            hdf5_dict = {}
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    key, value = line.rstrip().split(None, 1)

                    if ":" not in value:
                        raise RuntimeError(
                            "scp file for hdf5 should be like: "
                            '"uttid filepath.h5:key": {}({})'.format(
                                line, self.filepath))
                    path, h5_key = value.split(":", 1)

                    hdf5_file = hdf5_dict.get(path)
                    if hdf5_file is None:
                        try:
                            hdf5_file = h5py.File(path, "r")
                        except Exception:
                            logging.error("Error when loading {}".format(path))
                            raise
                        hdf5_dict[path] = hdf5_file

                    try:
                        data = hdf5_file[h5_key]
                    except Exception:
                        logging.error("Error when loading {} with key={}".
                                      format(path, h5_key))
                        raise

                    if self.return_shape:
                        yield key, data.shape
                    else:
                        yield key, data[()]

            # Closing all files
            for k in hdf5_dict:
                try:
                    hdf5_dict[k].close()
                except Exception:
                    pass

        else:
            if self.filepath == "-":
                # Required h5py>=2.9
                filepath = io.BytesIO(sys.stdin.buffer.read())
            else:
                filepath = self.filepath
            with h5py.File(filepath, "r") as f:
                for key in f:
                    if self.return_shape:
                        yield key, f[key].shape
                    else:
                        yield key, f[key][()]


class SoundHDF5Reader:
    def __init__(self, rspecifier, return_shape=False):
        if ":" not in rspecifier:
            raise ValueError('Give "rspecifier" such as "ark:some.ark: {}"'.
                             format(rspecifier))
        self.ark_or_scp, self.filepath = rspecifier.split(":", 1)
        if self.ark_or_scp not in ["ark", "scp"]:
            raise ValueError(f"Must be scp or ark: {self.ark_or_scp}")
        self.return_shape = return_shape

    def __iter__(self):
        if self.ark_or_scp == "scp":
            hdf5_dict = {}
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    key, value = line.rstrip().split(None, 1)

                    if ":" not in value:
                        raise RuntimeError(
                            "scp file for hdf5 should be like: "
                            '"uttid filepath.h5:key": {}({})'.format(
                                line, self.filepath))
                    path, h5_key = value.split(":", 1)

                    hdf5_file = hdf5_dict.get(path)
                    if hdf5_file is None:
                        try:
                            hdf5_file = SoundHDF5File(path, "r")
                        except Exception:
                            logging.error("Error when loading {}".format(path))
                            raise
                        hdf5_dict[path] = hdf5_file

                    try:
                        data = hdf5_file[h5_key]
                    except Exception:
                        logging.error("Error when loading {} with key={}".
                                      format(path, h5_key))
                        raise

                    # Change Tuple[ndarray, int] -> Tuple[int, ndarray]
                    # (soundfile style -> scipy style)
                    array, rate = data
                    if self.return_shape:
                        array = array.shape
                    yield key, (rate, array)

            # Closing all files
            for k in hdf5_dict:
                try:
                    hdf5_dict[k].close()
                except Exception:
                    pass

        else:
            if self.filepath == "-":
                # Required h5py>=2.9
                filepath = io.BytesIO(sys.stdin.buffer.read())
            else:
                filepath = self.filepath
            for key, (a, r) in SoundHDF5File(filepath, "r").items():
                if self.return_shape:
                    a = a.shape
                yield key, (r, a)


class SoundReader:
    def __init__(self, rspecifier, return_shape=False):
        if ":" not in rspecifier:
            raise ValueError('Give "rspecifier" such as "scp:some.scp: {}"'.
                             format(rspecifier))
        self.ark_or_scp, self.filepath = rspecifier.split(":", 1)
        if self.ark_or_scp != "scp":
            raise ValueError('Only supporting "scp" for sound file: {}'.format(
                self.ark_or_scp))
        self.return_shape = return_shape

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                key, sound_file_path = line.rstrip().split(None, 1)
                # Assume PCM16
                array, rate = soundfile.read(sound_file_path, dtype="int16")
                # Change Tuple[ndarray, int] -> Tuple[int, ndarray]
                # (soundfile style -> scipy style)
                if self.return_shape:
                    array = array.shape
                yield key, (rate, array)
