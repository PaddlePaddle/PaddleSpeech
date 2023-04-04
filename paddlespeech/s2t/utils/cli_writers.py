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
from pathlib import Path
from typing import Dict

import h5py
import kaldiio
import numpy
import soundfile

from paddlespeech.s2t.io.reader import SoundHDF5File
from paddlespeech.s2t.utils.cli_utils import assert_scipy_wav_style


def file_writer_helper(
    wspecifier: str,
    filetype: str = "mat",
    write_num_frames: str = None,
    compress: bool = False,
    compression_method: int = 2,
    pcm_format: str = "wav",
):
    """Write matrices in kaldi style

    Args:
        wspecifier: e.g. ark,scp:out.ark,out.scp
        filetype: "mat" is kaldi-martix, "hdf5": HDF5
        write_num_frames: e.g. 'ark,t:num_frames.txt'
        compress: Compress or not
        compression_method: Specify compression level

    Write in kaldi-matrix-ark with "kaldi-scp" file:

    >>> with file_writer_helper('ark,scp:out.ark,out.scp') as f:
    >>>     f['uttid'] = array

    This "scp" has the following format:

        uttidA out.ark:1234
        uttidB out.ark:2222

    where, 1234 and 2222 points the strating byte address of the matrix.
    (For detail, see official documentation of Kaldi)

    Write in HDF5 with "scp" file:

    >>> with file_writer_helper('ark,scp:out.h5,out.scp', 'hdf5') as f:
    >>>     f['uttid'] = array

    This "scp" file is created as:

        uttidA out.h5:uttidA
        uttidB out.h5:uttidB

    HDF5 can be, unlike "kaldi-ark", accessed to any keys,
    so originally "scp" is not required for random-reading.
    Nevertheless we create "scp" for HDF5 because it is useful
    for some use-case. e.g. Concatenation, Splitting.

    """
    if filetype == "mat":
        return KaldiWriter(
            wspecifier,
            write_num_frames=write_num_frames,
            compress=compress,
            compression_method=compression_method,
        )
    elif filetype == "hdf5":
        return HDF5Writer(wspecifier,
                          write_num_frames=write_num_frames,
                          compress=compress)
    elif filetype == "sound.hdf5":
        return SoundHDF5Writer(wspecifier,
                               write_num_frames=write_num_frames,
                               pcm_format=pcm_format)
    elif filetype == "sound":
        return SoundWriter(wspecifier,
                           write_num_frames=write_num_frames,
                           pcm_format=pcm_format)
    else:
        raise NotImplementedError(f"filetype={filetype}")


class BaseWriter:
    def __setitem__(self, key, value):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass

        if self.writer_scp is not None:
            try:
                self.writer_scp.close()
            except Exception:
                pass

        if self.writer_nframe is not None:
            try:
                self.writer_nframe.close()
            except Exception:
                pass


def get_num_frames_writer(write_num_frames: str):
    """get_num_frames_writer

    Examples:
        >>> get_num_frames_writer('ark,t:num_frames.txt')
    """
    if write_num_frames is not None:
        if ":" not in write_num_frames:
            raise ValueError('Must include ":", write_num_frames={}'.format(
                write_num_frames))

        nframes_type, nframes_file = write_num_frames.split(":", 1)
        if nframes_type != "ark,t":
            raise ValueError("Only supporting text mode. "
                             "e.g. --write-num-frames=ark,t:foo.txt :"
                             "{}".format(nframes_type))

    return open(nframes_file, "w", encoding="utf-8")


class KaldiWriter(BaseWriter):
    def __init__(self,
                 wspecifier,
                 write_num_frames=None,
                 compress=False,
                 compression_method=2):
        if compress:
            self.writer = kaldiio.WriteHelper(
                wspecifier, compression_method=compression_method)
        else:
            self.writer = kaldiio.WriteHelper(wspecifier)
        self.writer_scp = None
        if write_num_frames is not None:
            self.writer_nframe = get_num_frames_writer(write_num_frames)
        else:
            self.writer_nframe = None

    def __setitem__(self, key, value):
        self.writer[key] = value
        if self.writer_nframe is not None:
            self.writer_nframe.write(f"{key} {len(value)}\n")


def parse_wspecifier(wspecifier: str) -> Dict[str, str]:
    """Parse wspecifier to dict

    Examples:
        >>> parse_wspecifier('ark,scp:out.ark,out.scp')
        {'ark': 'out.ark', 'scp': 'out.scp'}

    """
    ark_scp, filepath = wspecifier.split(":", 1)
    if ark_scp not in ["ark", "scp,ark", "ark,scp"]:
        raise ValueError("{} is not allowed: {}".format(ark_scp, wspecifier))
    ark_scps = ark_scp.split(",")
    filepaths = filepath.split(",")
    if len(ark_scps) != len(filepaths):
        raise ValueError("Mismatch: {} and {}".format(ark_scp, filepath))
    spec_dict = dict(zip(ark_scps, filepaths))
    return spec_dict


class HDF5Writer(BaseWriter):
    """HDF5Writer

    Examples:
        >>> with HDF5Writer('ark:out.h5', compress=True) as f:
        ...     f['key'] = array
    """
    def __init__(self, wspecifier, write_num_frames=None, compress=False):
        spec_dict = parse_wspecifier(wspecifier)
        self.filename = spec_dict["ark"]

        if compress:
            self.kwargs = {"compression": "gzip"}
        else:
            self.kwargs = {}
        self.writer = h5py.File(spec_dict["ark"], "w")
        if "scp" in spec_dict:
            self.writer_scp = open(spec_dict["scp"], "w", encoding="utf-8")
        else:
            self.writer_scp = None
        if write_num_frames is not None:
            self.writer_nframe = get_num_frames_writer(write_num_frames)
        else:
            self.writer_nframe = None

    def __setitem__(self, key, value):
        self.writer.create_dataset(key, data=value, **self.kwargs)

        if self.writer_scp is not None:
            self.writer_scp.write(f"{key} {self.filename}:{key}\n")
        if self.writer_nframe is not None:
            self.writer_nframe.write(f"{key} {len(value)}\n")


class SoundHDF5Writer(BaseWriter):
    """SoundHDF5Writer

    Examples:
        >>> fs = 16000
        >>> with SoundHDF5Writer('ark:out.h5') as f:
        ...     f['key'] = fs, array
    """
    def __init__(self, wspecifier, write_num_frames=None, pcm_format="wav"):
        self.pcm_format = pcm_format
        spec_dict = parse_wspecifier(wspecifier)
        self.filename = spec_dict["ark"]
        self.writer = SoundHDF5File(spec_dict["ark"],
                                    "w",
                                    format=self.pcm_format)
        if "scp" in spec_dict:
            self.writer_scp = open(spec_dict["scp"], "w", encoding="utf-8")
        else:
            self.writer_scp = None
        if write_num_frames is not None:
            self.writer_nframe = get_num_frames_writer(write_num_frames)
        else:
            self.writer_nframe = None

    def __setitem__(self, key, value):
        assert_scipy_wav_style(value)
        # Change Tuple[int, ndarray] -> Tuple[ndarray, int]
        # (scipy style -> soundfile style)
        value = (value[1], value[0])
        self.writer.create_dataset(key, data=value)

        if self.writer_scp is not None:
            self.writer_scp.write(f"{key} {self.filename}:{key}\n")
        if self.writer_nframe is not None:
            self.writer_nframe.write(f"{key} {len(value[0])}\n")


class SoundWriter(BaseWriter):
    """SoundWriter

    Examples:
        >>> fs = 16000
        >>> with SoundWriter('ark,scp:outdir,out.scp') as f:
        ...     f['key'] = fs, array
    """
    def __init__(self, wspecifier, write_num_frames=None, pcm_format="wav"):
        self.pcm_format = pcm_format
        spec_dict = parse_wspecifier(wspecifier)
        # e.g. ark,scp:dirname,wav.scp
        # -> The wave files are found in dirname/*.wav
        self.dirname = spec_dict["ark"]
        Path(self.dirname).mkdir(parents=True, exist_ok=True)
        self.writer = None

        if "scp" in spec_dict:
            self.writer_scp = open(spec_dict["scp"], "w", encoding="utf-8")
        else:
            self.writer_scp = None
        if write_num_frames is not None:
            self.writer_nframe = get_num_frames_writer(write_num_frames)
        else:
            self.writer_nframe = None

    def __setitem__(self, key, value):
        assert_scipy_wav_style(value)
        rate, signal = value
        wavfile = Path(self.dirname) / (key + "." + self.pcm_format)
        soundfile.write(wavfile, signal.astype(numpy.int16), rate)

        if self.writer_scp is not None:
            self.writer_scp.write(f"{key} {wavfile}\n")
        if self.writer_nframe is not None:
            self.writer_nframe.write(f"{key} {len(signal)}\n")
