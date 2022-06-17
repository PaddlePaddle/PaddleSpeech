from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import paddle
import yaml
from sedit_arg_parser import parse_args
from yacs.config import CfgNode

from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_voc_inference


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


def eval_durs(phns, target_lang="chinese", fs=24000, hop_length=300):
    args = parse_args()

    if target_lang == 'english':
        args.am = "fastspeech2_ljspeech"
        args.am_config = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/default.yaml"
        args.am_ckpt = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/snapshot_iter_100000.pdz"
        args.am_stat = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/speech_stats.npy"
        args.phones_dict = "download/fastspeech2_nosil_ljspeech_ckpt_0.5/phone_id_map.txt"

    elif target_lang == 'chinese':
        args.am = "fastspeech2_csmsc"
        args.am_config = "download/fastspeech2_conformer_baker_ckpt_0.5/conformer.yaml"
        args.am_ckpt = "download/fastspeech2_conformer_baker_ckpt_0.5/snapshot_iter_76000.pdz"
        args.am_stat = "download/fastspeech2_conformer_baker_ckpt_0.5/speech_stats.npy"
        args.phones_dict = "download/fastspeech2_conformer_baker_ckpt_0.5/phone_id_map.txt"

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    # Init body.
    with open(args.am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))

    am_inference, am = get_am_inference(
        am=args.am,
        am_config=am_config,
        am_ckpt=args.am_ckpt,
        am_stat=args.am_stat,
        phones_dict=args.phones_dict,
        tones_dict=args.tones_dict,
        speaker_dict=args.speaker_dict,
        return_am=True)

    vocab_phones = {}
    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    for tone, id in phn_id:
        vocab_phones[tone] = int(id)
    vocab_size = len(vocab_phones)
    phonemes = [phn if phn in vocab_phones else "sp" for phn in phns]

    phone_ids = [vocab_phones[item] for item in phonemes]
    phone_ids.append(vocab_size - 1)
    phone_ids = paddle.to_tensor(np.array(phone_ids, np.int64))
    _, d_outs, _, _ = am.inference(phone_ids, spk_id=None, spk_emb=None)
    pre_d_outs = d_outs
    phu_durs_new = pre_d_outs * hop_length / fs
    phu_durs_new = phu_durs_new.tolist()[:-1]
    return phu_durs_new
