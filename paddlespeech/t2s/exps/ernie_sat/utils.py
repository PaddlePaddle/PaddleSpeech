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
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union
import os

import numpy as np
import paddle
import yaml
from yacs.config import CfgNode
import hashlib


from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_voc_inference

def _get_user():
    return os.path.expanduser('~').split('/')[-1]

def str2md5(string):
    md5_val = hashlib.md5(string.encode('utf8')).hexdigest()
    return md5_val

def get_tmp_name(text:str):
    return _get_user() + '_' + str(os.getpid()) + '_' + str2md5(text)

def get_dict(dictfile: str):
    word2phns_dict = {}
    with open(dictfile, 'r') as fid:
        for line in fid:
            line_lst = line.split()
            word, phn_lst = line_lst[0], line.split()[1:]
            if word not in word2phns_dict.keys():
                word2phns_dict[word] = ' '.join(phn_lst)
    return word2phns_dict


# 获取需要被 mask 的 mel 帧的范围
def get_span_bdy(mfa_start: List[float],
                 mfa_end: List[float],
                 span_to_repl: List[List[int]]):
    if span_to_repl[0] >= len(mfa_start):
        span_bdy = [mfa_end[-1], mfa_end[-1]]
    else:
        span_bdy = [mfa_start[span_to_repl[0]], mfa_end[span_to_repl[1] - 1]]
    return span_bdy


# mfa 获得的 duration 和 fs2 的 duration_predictor 获取的 duration 可能不同
# 此处获得一个缩放比例, 用于预测值和真实值之间的缩放
def get_dur_adj_factor(orig_dur: List[int],
                       pred_dur: List[int],
                       phns: List[str]):
    length = 0
    factor_list = []
    for orig, pred, phn in zip(orig_dur, pred_dur, phns):
        if pred == 0 or phn == 'sp':
            continue
        else:
            factor_list.append(orig / pred)
    factor_list = np.array(factor_list)
    factor_list.sort()
    if len(factor_list) < 5:
        return 1
    length = 2
    avg = np.average(factor_list[length:-length])
    return avg


def read_2col_text(path: Union[Path, str]) -> Dict[str, str]:
    """Read a text file having 2 column as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2col_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """

    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v
    return data


def load_num_sequence_text(path: Union[Path, str], loader_type: str="csv_int"
                           ) -> Dict[str, List[Union[float, int]]]:
    """Read a text file indicating sequences of number

    Examples:
        key1 1 2 3
        key2 34 5 6

        >>> d = load_num_sequence_text('text')
        >>> np.testing.assert_array_equal(d["key1"], np.array([1, 2, 3]))
    """
    if loader_type == "text_int":
        delimiter = " "
        dtype = int
    elif loader_type == "text_float":
        delimiter = " "
        dtype = float
    elif loader_type == "csv_int":
        delimiter = ","
        dtype = int
    elif loader_type == "csv_float":
        delimiter = ","
        dtype = float
    else:
        raise ValueError(f"Not supported loader_type={loader_type}")

    # path looks like:
    #   utta 1,0
    #   uttb 3,4,5
    # -> return {'utta': np.ndarray([1, 0]),
    #            'uttb': np.ndarray([3, 4, 5])}
    d = read_2column_text(path)
    # Using for-loop instead of dict-comprehension for debuggability
    retval = {}
    for k, v in d.items():
        try:
            retval[k] = [dtype(i) for i in v.split(delimiter)]
        except TypeError:
            print(f'Error happened with path="{path}", id="{k}", value="{v}"')
            raise
    return retval


def is_chinese(ch):
    if u'\u4e00' <= ch <= u'\u9fff':
        return True
    else:
        return False


def get_voc_out(mel):
    # vocoder
    args = parse_args()
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))
    voc_inference = get_voc_inference(
        voc=args.voc,
        voc_config=voc_config,
        voc_ckpt=args.voc_ckpt,
        voc_stat=args.voc_stat)

    with paddle.no_grad():
        wav = voc_inference(mel)
    return np.squeeze(wav)


def eval_durs(phns, target_lang: str='zh', fs: int=24000, n_shift: int=300):

    if target_lang == 'en':
        am = "fastspeech2_ljspeech"
        am_config = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/default.yaml"
        am_ckpt = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/snapshot_iter_100000.pdz"
        am_stat = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/speech_stats.npy"
        phones_dict = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/phone_id_map.txt"

    elif target_lang == 'zh':
        am = "fastspeech2_csmsc"
        am_config = "download/fastspeech2_conformer_baker_ckpt_0.5/conformer.yaml"
        am_ckpt = "download/fastspeech2_conformer_baker_ckpt_0.5/snapshot_iter_76000.pdz"
        am_stat = "download/fastspeech2_conformer_baker_ckpt_0.5/speech_stats.npy"
        phones_dict = "download/fastspeech2_conformer_baker_ckpt_0.5/phone_id_map.txt"

    # Init body.
    with open(am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))

    am_inference, am = get_am_inference(
        am=am,
        am_config=am_config,
        am_ckpt=am_ckpt,
        am_stat=am_stat,
        phones_dict=phones_dict,
        return_am=True)

    vocab_phones = {}
    with open(phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    for tone, id in phn_id:
        vocab_phones[tone] = int(id)
    vocab_size = len(vocab_phones)
    phonemes = [phn if phn in vocab_phones else "sp" for phn in phns]

    phone_ids = [vocab_phones[item] for item in phonemes]
    phone_ids = paddle.to_tensor(np.array(phone_ids, np.int64))
    _, d_outs, _, _ = am.inference(phone_ids)
    d_outs = d_outs.tolist()
    return d_outs
