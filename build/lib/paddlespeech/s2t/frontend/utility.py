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
"""Contains data helper functions."""
import json
import math
import tarfile
from collections import namedtuple
from typing import List
from typing import Optional
from typing import Text

import jsonlines
import numpy as np

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = [
    "load_dict", "load_cmvn", "read_manifest", "rms_to_db", "rms_to_dbfs",
    "max_dbfs", "mean_dbfs", "gain_db_to_ratio", "normalize_audio", "SOS",
    "EOS", "UNK", "BLANK", "MASKCTC", "SPACE", "convert_samples_to_float32",
    "convert_samples_from_float32"
]

IGNORE_ID = -1
# `sos` and `eos` using same token
SOS = "<eos>"
EOS = SOS
UNK = "<unk>"
BLANK = "<blank>"
MASKCTC = "<mask>"
SPACE = "<space>"


def load_dict(dict_path: Optional[Text], maskctc=False) -> Optional[List[Text]]:
    if dict_path is None:
        return None

    with open(dict_path, "r") as f:
        dictionary = f.readlines()
    # first token is `<blank>`
    # multi line: `<blank> 0\n`
    # one line: `<blank>`
    # space is relpace with <space>
    char_list = [entry[:-1].split(" ")[0] for entry in dictionary]
    if BLANK not in char_list:
        char_list.insert(0, BLANK)
    if EOS not in char_list:
        char_list.append(EOS)
    # for non-autoregressive maskctc model
    if maskctc and MASKCTC not in char_list:
        char_list.append(MASKCTC)
    return char_list


def read_manifest(
        manifest_path,
        max_input_len=float('inf'),
        min_input_len=0.0,
        max_output_len=float('inf'),
        min_output_len=0.0,
        max_output_input_ratio=float('inf'),
        min_output_input_ratio=0.0, ):
    """Load and parse manifest file.

    Args:
        manifest_path ([type]): Manifest file to load and parse.
        max_input_len ([type], optional): maximum output seq length,
            in seconds for raw wav, in frame numbers for feature data.
            Defaults to float('inf').
        min_input_len (float, optional): minimum input seq length,
            in seconds for raw wav, in frame numbers for feature data.
            Defaults to 0.0.
        max_output_len (float, optional): maximum input seq length,
            in modeling units. Defaults to 500.0.
        min_output_len (float, optional): minimum input seq length,
            in modeling units. Defaults to 0.0.
        max_output_input_ratio (float, optional):
            maximum output seq length/output seq length ratio. Defaults to 10.0.
        min_output_input_ratio (float, optional):
            minimum output seq length/output seq length ratio. Defaults to 0.05.

    Raises:
        IOError: If failed to parse the manifest.

    Returns:
        List[dict]: Manifest parsing results.
    """
    manifest = []
    with jsonlines.open(manifest_path, 'r') as reader:
        for json_data in reader:
            feat_len = json_data["input"][0]["shape"][
                0] if "input" in json_data and "shape" in json_data["input"][
                    0] else 1.0
            token_len = json_data["output"][0]["shape"][
                0] if "output" in json_data and "shape" in json_data["output"][
                    0] else 1.0
            conditions = [
                feat_len >= min_input_len,
                feat_len <= max_input_len,
                token_len >= min_output_len,
                token_len <= max_output_len,
                token_len / feat_len >= min_output_input_ratio,
                token_len / feat_len <= max_output_input_ratio,
            ]
            if all(conditions):
                manifest.append(json_data)
    return manifest


# Tar File read
TarLocalData = namedtuple('TarLocalData', ['tar2info', 'tar2object'])


def parse_tar(file):
    """Parse a tar file to get a tarfile object
    and a map containing tarinfoes
    """
    result = {}
    f = tarfile.open(file)
    for tarinfo in f.getmembers():
        result[tarinfo.name] = tarinfo
    return f, result


def subfile_from_tar(file, local_data=None):
    """Get subfile object from tar.

    tar:tarpath#filename

    It will return a subfile object from tar file
    and cached tar file info for next reading request.
    """
    tarpath, filename = file.split(':', 1)[1].split('#', 1)

    if local_data is None:
        local_data = TarLocalData(tar2info={}, tar2object={})

    assert isinstance(local_data, TarLocalData)

    if 'tar2info' not in local_data.__dict__:
        local_data.tar2info = {}
    if 'tar2object' not in local_data.__dict__:
        local_data.tar2object = {}

    if tarpath not in local_data.tar2info:
        fobj, infos = parse_tar(tarpath)
        local_data.tar2info[tarpath] = infos
        local_data.tar2object[tarpath] = fobj
    else:
        fobj = local_data.tar2object[tarpath]
        infos = local_data.tar2info[tarpath]
    return fobj.extractfile(infos[filename])


def rms_to_db(rms: float):
    """Root Mean Square to dB.

    Args:
        rms ([float]): root mean square

    Returns:
        float: dB
    """
    return 20.0 * math.log10(max(1e-16, rms))


def rms_to_dbfs(rms: float):
    """Root Mean Square to dBFS.
    https://fireattack.wordpress.com/2017/02/06/replaygain-loudness-normalization-and-applications/
    Audio is mix of sine wave, so 1 amp sine wave's Full scale is 0.7071, equal to -3.0103dB.

    dB = dBFS + 3.0103
    dBFS = db - 3.0103
    e.g. 0 dB = -3.0103 dBFS

    Args:
        rms ([float]): root mean square

    Returns:
        float: dBFS
    """
    return rms_to_db(rms) - 3.0103


def max_dbfs(sample_data: np.ndarray):
    """Peak dBFS based on the maximum energy sample.

    Args:
        sample_data ([np.ndarray]): float array, [-1, 1].

    Returns:
        float: dBFS
    """
    # Peak dBFS based on the maximum energy sample. Will prevent overdrive if used for normalization.
    return rms_to_dbfs(max(abs(np.min(sample_data)), abs(np.max(sample_data))))


def mean_dbfs(sample_data):
    """Peak dBFS based on the RMS energy.

    Args:
        sample_data ([np.ndarray]): float array, [-1, 1].

    Returns:
        float: dBFS
    """
    return rms_to_dbfs(
        math.sqrt(np.mean(np.square(sample_data, dtype=np.float64))))


def gain_db_to_ratio(gain_db: float):
    """dB to ratio

    Args:
        gain_db (float): gain in dB

    Returns:
        float: scale in amp
    """
    return math.pow(10.0, gain_db / 20.0)


def normalize_audio(sample_data: np.ndarray, dbfs: float=-3.0103):
    """Nomalize audio to dBFS.

    Args:
        sample_data (np.ndarray): input wave samples, [-1, 1].
        dbfs (float, optional): target dBFS. Defaults to -3.0103.

    Returns:
        np.ndarray: normalized wave
    """
    return np.maximum(
        np.minimum(sample_data * gain_db_to_ratio(dbfs - max_dbfs(sample_data)),
                   1.0), -1.0)


def _load_json_cmvn(json_cmvn_file):
    """ Load the json format cmvn stats file and calculate cmvn

    Args:
        json_cmvn_file: cmvn stats file in json format

    Returns:
        a numpy array of [means, vars]
    """
    with open(json_cmvn_file) as f:
        cmvn_stats = json.load(f)

    means = cmvn_stats['mean_stat']
    variance = cmvn_stats['var_stat']
    count = cmvn_stats['frame_num']
    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    cmvn = np.array([means, variance])
    return cmvn


def _load_kaldi_cmvn(kaldi_cmvn_file):
    """ Load the kaldi format cmvn stats file and calculate cmvn

    Args:
        kaldi_cmvn_file:  kaldi text style global cmvn file, which
           is generated by:
           compute-cmvn-stats --binary=false scp:feats.scp global_cmvn

    Returns:
        a numpy array of [means, vars]
    """
    means = []
    variance = []
    with open(kaldi_cmvn_file, 'r') as fid:
        # kaldi binary file start with '\0B'
        if fid.read(2) == '\0B':
            logger.error('kaldi cmvn binary file is not supported, please '
                         'recompute it by: compute-cmvn-stats --binary=false '
                         ' scp:feats.scp global_cmvn')
            sys.exit(1)
        fid.seek(0)
        arr = fid.read().split()
        assert (arr[0] == '[')
        assert (arr[-2] == '0')
        assert (arr[-1] == ']')
        feat_dim = int((len(arr) - 2 - 2) / 2)
        for i in range(1, feat_dim + 1):
            means.append(float(arr[i]))
        count = float(arr[feat_dim + 1])
        for i in range(feat_dim + 2, 2 * feat_dim + 2):
            variance.append(float(arr[i]))

    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    cmvn = np.array([means, variance])
    return cmvn


def load_cmvn(cmvn_file: str, filetype: str):
    """load cmvn from file.

    Args:
        cmvn_file (str): cmvn path.
        filetype (str): file type, optional[npz, json, kaldi].

    Raises:
        ValueError: file type not support.

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, istd
    """
    assert filetype in ['npz', 'json', 'kaldi'], filetype
    filetype = filetype.lower()
    if filetype == "json":
        cmvn = _load_json_cmvn(cmvn_file)
    elif filetype == "kaldi":
        cmvn = _load_kaldi_cmvn(cmvn_file)
    elif filetype == "npz":
        eps = 1e-14
        npzfile = np.load(cmvn_file)
        mean = np.squeeze(npzfile["mean"])
        std = np.squeeze(npzfile["std"])
        istd = 1 / (std + eps)
        cmvn = [mean, istd]
    else:
        raise ValueError(f"cmvn file type no support: {filetype}")
    return cmvn[0], cmvn[1]


def convert_samples_to_float32(samples):
    """Convert sample type to float32.

    Audio sample type is usually integer or float-point.
    Integers will be scaled to [-1, 1] in float32.

    PCM16 -> PCM32
    """
    float32_samples = samples.astype('float32')
    if samples.dtype in np.sctypes['int']:
        bits = np.iinfo(samples.dtype).bits
        float32_samples *= (1. / 2**(bits - 1))
    elif samples.dtype in np.sctypes['float']:
        pass
    else:
        raise TypeError("Unsupported sample type: %s." % samples.dtype)
    return float32_samples


def convert_samples_from_float32(samples, dtype):
    """Convert sample type from float32 to dtype.

    Audio sample type is usually integer or float-point. For integer
    type, float32 will be rescaled from [-1, 1] to the maximum range
    supported by the integer type.

    PCM32 -> PCM16
    """
    dtype = np.dtype(dtype)
    output_samples = samples.copy()
    if dtype in np.sctypes['int']:
        bits = np.iinfo(dtype).bits
        output_samples *= (2**(bits - 1) / 1.)
        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max
        output_samples[output_samples > max_val] = max_val
        output_samples[output_samples < min_val] = min_val
    elif samples.dtype in np.sctypes['float']:
        min_val = np.finfo(dtype).min
        max_val = np.finfo(dtype).max
        output_samples[output_samples > max_val] = max_val
        output_samples[output_samples < min_val] = min_val
    else:
        raise TypeError("Unsupported sample type: %s." % samples.dtype)
    return output_samples.astype(dtype)
