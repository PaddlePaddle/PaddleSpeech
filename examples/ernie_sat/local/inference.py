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
import os
import random
from typing import Dict
from typing import List

import librosa
import numpy as np
import paddle
import soundfile as sf
from align import alignment
from align import alignment_zh
from align import words2phns
from align import words2phns_zh
from paddle import nn
from sedit_arg_parser import parse_args
from utils import eval_durs
from utils import get_voc_out
from utils import is_chinese
from utils import load_num_sequence_text
from utils import read_2col_text

from paddlespeech.t2s.datasets.am_batch_fn import build_mlm_collate_fn
from paddlespeech.t2s.models.ernie_sat.mlm import build_model_from_file

random.seed(0)
np.random.seed(0)


def get_wav(wav_path: str,
            source_lang: str='english',
            target_lang: str='english',
            model_name: str="paddle_checkpoint_en",
            old_str: str="",
            new_str: str="",
            non_autoreg: bool=True):
    wav_org, output_feat, old_span_bdy, new_span_bdy, fs, hop_length = get_mlm_output(
        source_lang=source_lang,
        target_lang=target_lang,
        model_name=model_name,
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        use_teacher_forcing=non_autoreg)

    masked_feat = output_feat[new_span_bdy[0]:new_span_bdy[1]]

    alt_wav = get_voc_out(masked_feat)

    old_time_bdy = [hop_length * x for x in old_span_bdy]

    wav_replaced = np.concatenate(
        [wav_org[:old_time_bdy[0]], alt_wav, wav_org[old_time_bdy[1]:]])

    data_dict = {"origin": wav_org, "output": wav_replaced}

    return data_dict


def load_model(model_name: str="paddle_checkpoint_en"):
    config_path = './pretrained_model/{}/config.yaml'.format(model_name)
    model_path = './pretrained_model/{}/model.pdparams'.format(model_name)
    mlm_model, conf = build_model_from_file(
        config_file=config_path, model_file=model_path)
    return mlm_model, conf


def read_data(uid: str, prefix: os.PathLike):
    # 获取 uid 对应的文本
    mfa_text = read_2col_text(prefix + '/text')[uid]
    # 获取 uid 对应的音频路径
    mfa_wav_path = read_2col_text(prefix + '/wav.scp')[uid]
    if not os.path.isabs(mfa_wav_path):
        mfa_wav_path = prefix + mfa_wav_path
    return mfa_text, mfa_wav_path


def get_align_data(uid: str, prefix: os.PathLike):
    mfa_path = prefix + "mfa_"
    mfa_text = read_2col_text(mfa_path + 'text')[uid]
    mfa_start = load_num_sequence_text(
        mfa_path + 'start', loader_type='text_float')[uid]
    mfa_end = load_num_sequence_text(
        mfa_path + 'end', loader_type='text_float')[uid]
    mfa_wav_path = read_2col_text(mfa_path + 'wav.scp')[uid]
    return mfa_text, mfa_start, mfa_end, mfa_wav_path


# 获取需要被 mask 的 mel 帧的范围
def get_masked_mel_bdy(mfa_start: List[float],
                       mfa_end: List[float],
                       fs: int,
                       hop_length: int,
                       span_to_repl: List[List[int]]):
    align_start = np.array(mfa_start)
    align_end = np.array(mfa_end)
    align_start = np.floor(fs * align_start / hop_length).astype('int')
    align_end = np.floor(fs * align_end / hop_length).astype('int')
    if span_to_repl[0] >= len(mfa_start):
        span_bdy = [align_end[-1], align_end[-1]]
    else:
        span_bdy = [
            align_start[span_to_repl[0]], align_end[span_to_repl[1] - 1]
        ]
    return span_bdy, align_start, align_end


def recover_dict(word2phns: Dict[str, str], tp_word2phns: Dict[str, str]):
    dic = {}
    keys_to_del = []
    exist_idx = []
    sp_count = 0
    add_sp_count = 0
    for key in word2phns.keys():
        idx, wrd = key.split('_')
        if wrd == 'sp':
            sp_count += 1
            exist_idx.append(int(idx))
        else:
            keys_to_del.append(key)

    for key in keys_to_del:
        del word2phns[key]

    cur_id = 0
    for key in tp_word2phns.keys():
        if cur_id in exist_idx:
            dic[str(cur_id) + "_sp"] = 'sp'
            cur_id += 1
            add_sp_count += 1
        idx, wrd = key.split('_')
        dic[str(cur_id) + "_" + wrd] = tp_word2phns[key]
        cur_id += 1

    if add_sp_count + 1 == sp_count:
        dic[str(cur_id) + "_sp"] = 'sp'
        add_sp_count += 1

    assert add_sp_count == sp_count, "sp are not added in dic"
    return dic


def get_max_idx(dic):
    return sorted([int(key.split('_')[0]) for key in dic.keys()])[-1]


def get_phns_and_spans(wav_path: str,
                       old_str: str="",
                       new_str: str="",
                       source_lang: str="english",
                       target_lang: str="english"):
    is_append = (old_str == new_str[:len(old_str)])
    old_phns, mfa_start, mfa_end = [], [], []
    # source
    if source_lang == "english":
        intervals, word2phns = alignment(wav_path, old_str)
    elif source_lang == "chinese":
        intervals, word2phns = alignment_zh(wav_path, old_str)
        _, tp_word2phns = words2phns_zh(old_str)

        for key, value in tp_word2phns.items():
            idx, wrd = key.split('_')
            cur_val = " ".join(value)
            tp_word2phns[key] = cur_val

        word2phns = recover_dict(word2phns, tp_word2phns)
    else:
        assert source_lang == "chinese" or source_lang == "english", \
            "source_lang is wrong..."

    for item in intervals:
        old_phns.append(item[0])
        mfa_start.append(float(item[1]))
        mfa_end.append(float(item[2]))
    # target
    if is_append and (source_lang != target_lang):
        cross_lingual_clone = True
    else:
        cross_lingual_clone = False

    if cross_lingual_clone:
        str_origin = new_str[:len(old_str)]
        str_append = new_str[len(old_str):]

        if target_lang == "chinese":
            phns_origin, origin_word2phns = words2phns(str_origin)
            phns_append, append_word2phns_tmp = words2phns_zh(str_append)

        elif target_lang == "english":
            # 原始句子
            phns_origin, origin_word2phns = words2phns_zh(str_origin)
            # clone 句子 
            phns_append, append_word2phns_tmp = words2phns(str_append)
        else:
            assert target_lang == "chinese" or target_lang == "english", \
                "cloning is not support for this language, please check it."

        new_phns = phns_origin + phns_append

        append_word2phns = {}
        length = len(origin_word2phns)
        for key, value in append_word2phns_tmp.items():
            idx, wrd = key.split('_')
            append_word2phns[str(int(idx) + length) + '_' + wrd] = value
        new_word2phns = origin_word2phns.copy()
        new_word2phns.update(append_word2phns)

    else:
        if source_lang == target_lang and target_lang == "english":
            new_phns, new_word2phns = words2phns(new_str)
        elif source_lang == target_lang and target_lang == "chinese":
            new_phns, new_word2phns = words2phns_zh(new_str)
        else:
            assert source_lang == target_lang, \
                "source language is not same with target language..."

    span_to_repl = [0, len(old_phns) - 1]
    span_to_add = [0, len(new_phns) - 1]
    left_idx = 0
    new_phns_left = []
    sp_count = 0
    # find the left different index
    for key in word2phns.keys():
        idx, wrd = key.split('_')
        if wrd == 'sp':
            sp_count += 1
            new_phns_left.append('sp')
        else:
            idx = str(int(idx) - sp_count)
            if idx + '_' + wrd in new_word2phns:
                left_idx += len(new_word2phns[idx + '_' + wrd])
                new_phns_left.extend(word2phns[key].split())
            else:
                span_to_repl[0] = len(new_phns_left)
                span_to_add[0] = len(new_phns_left)
                break

    # reverse word2phns and new_word2phns
    right_idx = 0
    new_phns_right = []
    sp_count = 0
    word2phns_max_idx = get_max_idx(word2phns)
    new_word2phns_max_idx = get_max_idx(new_word2phns)
    new_phns_mid = []
    if is_append:
        new_phns_right = []
        new_phns_mid = new_phns[left_idx:]
        span_to_repl[0] = len(new_phns_left)
        span_to_add[0] = len(new_phns_left)
        span_to_add[1] = len(new_phns_left) + len(new_phns_mid)
        span_to_repl[1] = len(old_phns) - len(new_phns_right)
    # speech edit
    else:
        for key in list(word2phns.keys())[::-1]:
            idx, wrd = key.split('_')
            if wrd == 'sp':
                sp_count += 1
                new_phns_right = ['sp'] + new_phns_right
            else:
                idx = str(new_word2phns_max_idx - (word2phns_max_idx - int(idx)
                                                   - sp_count))
                if idx + '_' + wrd in new_word2phns:
                    right_idx -= len(new_word2phns[idx + '_' + wrd])
                    new_phns_right = word2phns[key].split() + new_phns_right
                else:
                    span_to_repl[1] = len(old_phns) - len(new_phns_right)
                    new_phns_mid = new_phns[left_idx:right_idx]
                    span_to_add[1] = len(new_phns_left) + len(new_phns_mid)
                    if len(new_phns_mid) == 0:
                        span_to_add[1] = min(span_to_add[1] + 1, len(new_phns))
                        span_to_add[0] = max(0, span_to_add[0] - 1)
                        span_to_repl[0] = max(0, span_to_repl[0] - 1)
                        span_to_repl[1] = min(span_to_repl[1] + 1,
                                              len(old_phns))
                    break
    new_phns = new_phns_left + new_phns_mid + new_phns_right
    '''
    For that reason cover should not be given.
    For that reason cover is impossible to be given.
    span_to_repl: [17, 23] "should not"
    span_to_add: [17, 30]  "is impossible to"
    '''
    return mfa_start, mfa_end, old_phns, new_phns, span_to_repl, span_to_add


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


def prep_feats_with_dur(wav_path: str,
                        source_lang: str="English",
                        target_lang: str="English",
                        old_str: str="",
                        new_str: str="",
                        mask_reconstruct: bool=False,
                        duration_adjust: bool=True,
                        start_end_sp: bool=False,
                        fs: int=24000,
                        hop_length: int=300):
    '''
    Returns:
        np.ndarray: new wav, replace the part to be edited in original wav with 0
        List[str]: new phones
        List[float]: mfa start of new wav
        List[float]: mfa end of new wav
        List[int]: masked mel boundary of original wav
        List[int]: masked mel boundary of new wav
    '''
    wav_org, _ = librosa.load(wav_path, sr=fs)

    mfa_start, mfa_end, old_phns, new_phns, span_to_repl, span_to_add = get_phns_and_spans(
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        source_lang=source_lang,
        target_lang=target_lang)

    if start_end_sp:
        if new_phns[-1] != 'sp':
            new_phns = new_phns + ['sp']
    # 中文的 phns 不一定都在 fastspeech2 的字典里, 用 sp 代替
    if target_lang == "english" or target_lang == "chinese":
        old_durs = eval_durs(old_phns, target_lang=source_lang)
    else:
        assert target_lang == "chinese" or target_lang == "english", \
            "calculate duration_predict is not support for this language..."

    orig_old_durs = [e - s for e, s in zip(mfa_end, mfa_start)]
    if '[MASK]' in new_str:
        new_phns = old_phns
        span_to_add = span_to_repl
        d_factor_left = get_dur_adj_factor(
            orig_dur=orig_old_durs[:span_to_repl[0]],
            pred_dur=old_durs[:span_to_repl[0]],
            phns=old_phns[:span_to_repl[0]])
        d_factor_right = get_dur_adj_factor(
            orig_dur=orig_old_durs[span_to_repl[1]:],
            pred_dur=old_durs[span_to_repl[1]:],
            phns=old_phns[span_to_repl[1]:])
        d_factor = (d_factor_left + d_factor_right) / 2
        new_durs_adjusted = [d_factor * i for i in old_durs]
    else:
        if duration_adjust:
            d_factor = get_dur_adj_factor(
                orig_dur=orig_old_durs, pred_dur=old_durs, phns=old_phns)
            d_factor = d_factor * 1.25
        else:
            d_factor = 1

        if target_lang == "english" or target_lang == "chinese":
            new_durs = eval_durs(new_phns, target_lang=target_lang)
        else:
            assert target_lang == "chinese" or target_lang == "english", \
                "calculate duration_predict is not support for this language..."

        new_durs_adjusted = [d_factor * i for i in new_durs]

    new_span_dur_sum = sum(new_durs_adjusted[span_to_add[0]:span_to_add[1]])
    old_span_dur_sum = sum(orig_old_durs[span_to_repl[0]:span_to_repl[1]])
    dur_offset = new_span_dur_sum - old_span_dur_sum
    new_mfa_start = mfa_start[:span_to_repl[0]]
    new_mfa_end = mfa_end[:span_to_repl[0]]
    for i in new_durs_adjusted[span_to_add[0]:span_to_add[1]]:
        if len(new_mfa_end) == 0:
            new_mfa_start.append(0)
            new_mfa_end.append(i)
        else:
            new_mfa_start.append(new_mfa_end[-1])
            new_mfa_end.append(new_mfa_end[-1] + i)
    new_mfa_start += [i + dur_offset for i in mfa_start[span_to_repl[1]:]]
    new_mfa_end += [i + dur_offset for i in mfa_end[span_to_repl[1]:]]

    # 3. get new wav
    # 在原始句子后拼接
    if span_to_repl[0] >= len(mfa_start):
        left_idx = len(wav_org)
        right_idx = left_idx
    # 在原始句子中间替换
    else:
        left_idx = int(np.floor(mfa_start[span_to_repl[0]] * fs))
        right_idx = int(np.ceil(mfa_end[span_to_repl[1] - 1] * fs))
    blank_wav = np.zeros(
        (int(np.ceil(new_span_dur_sum * fs)), ), dtype=wav_org.dtype)
    # 原始音频，需要编辑的部分替换成空音频，空音频的时间由 fs2 的 duration_predictor 决定
    new_wav = np.concatenate(
        [wav_org[:left_idx], blank_wav, wav_org[right_idx:]])

    # 4. get old and new mel span to be mask
    # [92, 92]

    old_span_bdy, mfa_start, mfa_end = get_masked_mel_bdy(
        mfa_start=mfa_start,
        mfa_end=mfa_end,
        fs=fs,
        hop_length=hop_length,
        span_to_repl=span_to_repl)
    # [92, 174]
    # new_mfa_start, new_mfa_end 时间级别的开始和结束时间 -> 帧级别
    new_span_bdy, new_mfa_start, new_mfa_end = get_masked_mel_bdy(
        mfa_start=new_mfa_start,
        mfa_end=new_mfa_end,
        fs=fs,
        hop_length=hop_length,
        span_to_repl=span_to_add)

    # old_span_bdy, new_span_bdy 是帧级别的范围
    return new_wav, new_phns, new_mfa_start, new_mfa_end, old_span_bdy, new_span_bdy


def prep_feats(wav_path: str,
               source_lang: str="english",
               target_lang: str="english",
               old_str: str="",
               new_str: str="",
               duration_adjust: bool=True,
               start_end_sp: bool=False,
               mask_reconstruct: bool=False,
               fs: int=24000,
               hop_length: int=300,
               token_list: List[str]=[]):
    wav, phns, mfa_start, mfa_end, old_span_bdy, new_span_bdy = prep_feats_with_dur(
        source_lang=source_lang,
        target_lang=target_lang,
        old_str=old_str,
        new_str=new_str,
        wav_path=wav_path,
        duration_adjust=duration_adjust,
        start_end_sp=start_end_sp,
        mask_reconstruct=mask_reconstruct,
        fs=fs,
        hop_length=hop_length)

    token_to_id = {item: i for i, item in enumerate(token_list)}
    text = np.array(
        list(map(lambda x: token_to_id.get(x, token_to_id['<unk>']), phns)))
    span_bdy = np.array(new_span_bdy)

    batch = [('1', {
        "speech": wav,
        "align_start": mfa_start,
        "align_end": mfa_end,
        "text": text,
        "span_bdy": span_bdy
    })]

    return batch, old_span_bdy, new_span_bdy


def decode_with_model(mlm_model: nn.Layer,
                      collate_fn,
                      wav_path: str,
                      source_lang: str="english",
                      target_lang: str="english",
                      old_str: str="",
                      new_str: str="",
                      use_teacher_forcing: bool=False,
                      duration_adjust: bool=True,
                      start_end_sp: bool=False,
                      fs: int=24000,
                      hop_length: int=300,
                      token_list: List[str]=[]):
    batch, old_span_bdy, new_span_bdy = prep_feats(
        source_lang=source_lang,
        target_lang=target_lang,
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        duration_adjust=duration_adjust,
        start_end_sp=start_end_sp,
        fs=fs,
        hop_length=hop_length,
        token_list=token_list)

    feats = collate_fn(batch)[1]

    if 'text_masked_pos' in feats.keys():
        feats.pop('text_masked_pos')

    output = mlm_model.inference(
        text=feats['text'],
        speech=feats['speech'],
        masked_pos=feats['masked_pos'],
        speech_mask=feats['speech_mask'],
        text_mask=feats['text_mask'],
        speech_seg_pos=feats['speech_seg_pos'],
        text_seg_pos=feats['text_seg_pos'],
        span_bdy=new_span_bdy,
        use_teacher_forcing=use_teacher_forcing)

    # 拼接音频
    output_feat = paddle.concat(x=output, axis=0)
    wav_org, _ = librosa.load(wav_path, sr=fs)
    return wav_org, output_feat, old_span_bdy, new_span_bdy, fs, hop_length


def get_mlm_output(wav_path: str,
                   model_name: str="paddle_checkpoint_en",
                   source_lang: str="english",
                   target_lang: str="english",
                   old_str: str="",
                   new_str: str="",
                   use_teacher_forcing: bool=False,
                   duration_adjust: bool=True,
                   start_end_sp: bool=False):
    mlm_model, train_conf = load_model(model_name)
    mlm_model.eval()

    collate_fn = build_mlm_collate_fn(
        sr=train_conf.feats_extract_conf['fs'],
        n_fft=train_conf.feats_extract_conf['n_fft'],
        hop_length=train_conf.feats_extract_conf['hop_length'],
        win_length=train_conf.feats_extract_conf['win_length'],
        n_mels=train_conf.feats_extract_conf['n_mels'],
        fmin=train_conf.feats_extract_conf['fmin'],
        fmax=train_conf.feats_extract_conf['fmax'],
        mlm_prob=train_conf['mlm_prob'],
        mean_phn_span=train_conf['mean_phn_span'],
        seg_emb=train_conf.encoder_conf['input_layer'] == 'sega_mlm')

    return decode_with_model(
        source_lang=source_lang,
        target_lang=target_lang,
        mlm_model=mlm_model,
        collate_fn=collate_fn,
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        use_teacher_forcing=use_teacher_forcing,
        duration_adjust=duration_adjust,
        start_end_sp=start_end_sp,
        fs=train_conf.feats_extract_conf['fs'],
        hop_length=train_conf.feats_extract_conf['hop_length'],
        token_list=train_conf.token_list)


def evaluate(uid: str,
             source_lang: str="english",
             target_lang: str="english",
             prefix: os.PathLike="./prompt/dev/",
             model_name: str="paddle_checkpoint_en",
             new_str: str="",
             prompt_decoding: bool=False,
             task_name: str=None):

    # get origin text and path of origin wav
    old_str, wav_path = read_data(uid=uid, prefix=prefix)

    if task_name == 'edit':
        new_str = new_str
    elif task_name == 'synthesize':
        new_str = old_str + new_str
    else:
        new_str = old_str + ' '.join([ch for ch in new_str if is_chinese(ch)])

    print('new_str is ', new_str)

    results_dict = get_wav(
        source_lang=source_lang,
        target_lang=target_lang,
        model_name=model_name,
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str)
    return results_dict


if __name__ == "__main__":
    # parse config and args
    args = parse_args()

    data_dict = evaluate(
        uid=args.uid,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        prefix=args.prefix,
        model_name=args.model_name,
        new_str=args.new_str,
        task_name=args.task_name)
    sf.write(args.output_name, data_dict['output'], samplerate=24000)
    print("finished...")
