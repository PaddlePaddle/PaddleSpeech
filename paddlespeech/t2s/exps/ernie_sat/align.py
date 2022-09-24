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
import shutil
from pathlib import Path

import librosa
import numpy as np
import pypinyin
from praatio import textgrid

from paddlespeech.t2s.exps.ernie_sat.utils import get_dict
from paddlespeech.t2s.exps.ernie_sat.utils import get_tmp_name

DICT_EN = 'tools/aligner/cmudict-0.7b'
DICT_ZH = 'tools/aligner/simple.lexicon'
MODEL_DIR_EN = 'tools/aligner/vctk_model.zip'
MODEL_DIR_ZH = 'tools/aligner/aishell3_model.zip'
MFA_PATH = 'tools/montreal-forced-aligner/bin'
os.environ['PATH'] = MFA_PATH + '/:' + os.environ['PATH']


def _get_max_idx(dic):
    return sorted([int(key.split('_')[0]) for key in dic.keys()])[-1]


def _readtg(tg_path: str, lang: str='en', fs: int=24000, n_shift: int=300):
    alignment = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
    phones = []
    ends = []
    words = []

    for interval in alignment.tierDict['words'].entryList:
        word = interval.label
        if word:
            words.append(word)
    for interval in alignment.tierDict['phones'].entryList:
        phone = interval.label
        phones.append(phone)
        ends.append(interval.end)
    frame_pos = librosa.time_to_frames(ends, sr=fs, hop_length=n_shift)
    durations = np.diff(frame_pos, prepend=0)
    assert len(durations) == len(phones)
    # merge '' and sp in the end
    if phones[-1] == '' and len(phones) > 1 and phones[-2] == 'sp':
        phones = phones[:-1]
        durations[-2] += durations[-1]
        durations = durations[:-1]

    # replace ' and 'sil' with 'sp'
    phones = ['sp' if (phn == '' or phn == 'sil') else phn for phn in phones]

    if lang == 'en':
        DICT = DICT_EN

    elif lang == 'zh':
        DICT = DICT_ZH

    word2phns_dict = get_dict(DICT)

    phn2word_dict = []
    for word in words:
        if lang == 'en':
            word = word.upper()
        phn2word_dict.append([word2phns_dict[word].split(), word])

    non_sp_idx = 0
    word_idx = 0
    i = 0
    word2phns = {}
    while i < len(phones):
        phn = phones[i]
        if phn == 'sp':
            word2phns[str(word_idx) + '_sp'] = ['sp']
            i += 1
        else:
            phns, word = phn2word_dict[non_sp_idx]
            word2phns[str(word_idx) + '_' + word] = phns
            non_sp_idx += 1
            i += len(phns)
        word_idx += 1
    sum_phn = sum(len(word2phns[k]) for k in word2phns)
    assert sum_phn == len(phones)

    results = ''
    for (p, d) in zip(phones, durations):
        results += p + ' ' + str(d) + ' '
    return results.strip(), word2phns


def alignment(wav_path: str,
              text: str,
              fs: int=24000,
              lang='en',
              n_shift: int=300):
    wav_name = os.path.basename(wav_path)
    utt = wav_name.split('.')[0]
    # prepare data for MFA
    tmp_name = get_tmp_name(text=text)
    tmpbase = './tmp_dir/' + tmp_name
    tmpbase = Path(tmpbase)
    tmpbase.mkdir(parents=True, exist_ok=True)
    print("tmp_name in alignment:", tmp_name)

    shutil.copyfile(wav_path, tmpbase / wav_name)
    txt_name = utt + '.txt'
    txt_path = tmpbase / txt_name
    with open(txt_path, 'w') as wf:
        wf.write(text + '\n')
    # MFA
    if lang == 'en':
        DICT = DICT_EN
        MODEL_DIR = MODEL_DIR_EN

    elif lang == 'zh':
        DICT = DICT_ZH
        MODEL_DIR = MODEL_DIR_ZH
    else:
        print('please input right lang!!')

    CMD = 'mfa_align' + ' ' + str(
        tmpbase) + ' ' + DICT + ' ' + MODEL_DIR + ' ' + str(tmpbase)
    os.system(CMD)
    tg_path = str(tmpbase) + '/' + tmp_name + '/' + utt + '.TextGrid'
    phn_dur, word2phns = _readtg(tg_path, lang=lang)
    phn_dur = phn_dur.split()
    phns = phn_dur[::2]
    durs = phn_dur[1::2]
    durs = [int(d) for d in durs]
    assert len(phns) == len(durs)
    return phns, durs, word2phns


def words2phns(text: str, lang='en'):
    '''
    Args:
        text (str): 
            input text.
            eg: for that reason cover is impossible to be given.
        lang (str):
            'en' or 'zh'
    Returns:
        List[str]: phones of input text.
            eg:
            ['F', 'AO1', 'R', 'DH', 'AE1', 'T', 'R', 'IY1', 'Z', 'AH0', 'N', 'K', 'AH1', 'V', 'ER0',
            'IH1', 'Z', 'IH2', 'M', 'P', 'AA1', 'S', 'AH0', 'B', 'AH0', 'L', 'T', 'UW1', 'B', 'IY1', 
            'G', 'IH1', 'V', 'AH0', 'N']

        Dict(str, str): key - idx_word
                        value - phones
            eg:
            {'0_FOR': ['F', 'AO1', 'R'], '1_THAT': ['DH', 'AE1', 'T'], 
            '2_REASON': ['R', 'IY1', 'Z', 'AH0', 'N'],'3_COVER': ['K', 'AH1', 'V', 'ER0'], '4_IS': ['IH1', 'Z'], 
            '5_IMPOSSIBLE': ['IH2', 'M', 'P', 'AA1', 'S', 'AH0', 'B', 'AH0', 'L'],
            '6_TO': ['T', 'UW1'], '7_BE': ['B', 'IY1'], '8_GIVEN': ['G', 'IH1', 'V', 'AH0', 'N']}
    '''
    text = text.strip()
    words = []
    for pun in [
            ',', '.', ':', ';', '!', '?', '"', '(', ')', '--', '---', u'，',
            u'。', u'：', u'；', u'！', u'？', u'（', u'）'
    ]:
        text = text.replace(pun, ' ')
    for wrd in text.split():
        if (wrd[-1] == '-'):
            wrd = wrd[:-1]
        if (wrd[0] == "'"):
            wrd = wrd[1:]
        if wrd:
            words.append(wrd)
    if lang == 'en':
        dictfile = DICT_EN
    elif lang == 'zh':
        dictfile = DICT_ZH
    else:
        print('please input right lang!!')

    word2phns_dict = get_dict(dictfile)
    ds = word2phns_dict.keys()
    phns = []
    wrd2phns = {}
    for index, wrd in enumerate(words):
        if lang == 'en':
            wrd = wrd.upper()
        if (wrd not in ds):
            wrd2phns[str(index) + '_' + wrd] = 'spn'
            phns.extend('spn')
        else:
            wrd2phns[str(index) + '_' + wrd] = word2phns_dict[wrd].split()
            phns.extend(word2phns_dict[wrd].split())
    return phns, wrd2phns


def get_phns_spans(wav_path: str,
                   old_str: str='',
                   new_str: str='',
                   source_lang: str='en',
                   target_lang: str='en',
                   fs: int=24000,
                   n_shift: int=300):
    is_append = (old_str == new_str[:len(old_str)])
    old_phns, mfa_start, mfa_end = [], [], []
    # source
    lang = source_lang
    phn, dur, w2p = alignment(
        wav_path=wav_path, text=old_str, lang=lang, fs=fs, n_shift=n_shift)

    new_d_cumsum = np.pad(np.array(dur).cumsum(0), (1, 0), 'constant').tolist()
    mfa_start = new_d_cumsum[:-1]
    mfa_end = new_d_cumsum[1:]
    old_phns = phn

    # target
    if is_append and (source_lang != target_lang):
        cross_lingual_clone = True
    else:
        cross_lingual_clone = False

    if cross_lingual_clone:
        str_origin = new_str[:len(old_str)]
        str_append = new_str[len(old_str):]

        if target_lang == 'zh':
            phns_origin, origin_w2p = words2phns(str_origin, lang='en')
            phns_append, append_w2p_tmp = words2phns(str_append, lang='zh')
        elif target_lang == 'en':
            # 原始句子
            phns_origin, origin_w2p = words2phns(str_origin, lang='zh')
            # clone 句子 
            phns_append, append_w2p_tmp = words2phns(str_append, lang='en')
        else:
            assert target_lang == 'zh' or target_lang == 'en', \
                'cloning is not support for this language, please check it.'

        new_phns = phns_origin + phns_append

        append_w2p = {}
        length = len(origin_w2p)
        for key, value in append_w2p_tmp.items():
            idx, wrd = key.split('_')
            append_w2p[str(int(idx) + length) + '_' + wrd] = value
        new_w2p = origin_w2p.copy()
        new_w2p.update(append_w2p)

    else:
        if source_lang == target_lang:
            new_phns, new_w2p = words2phns(new_str, lang=source_lang)
        else:
            assert source_lang == target_lang, \
                'source language is not same with target language...'

    span_to_repl = [0, len(old_phns) - 1]
    span_to_add = [0, len(new_phns) - 1]
    left_idx = 0
    new_phns_left = []
    sp_count = 0
    # find the left different index
    # 因为可能 align 时候的 words2phns 和直接 words2phns, 前者会有 sp？
    for key in w2p.keys():
        idx, wrd = key.split('_')
        if wrd == 'sp':
            sp_count += 1
            new_phns_left.append('sp')
        else:
            idx = str(int(idx) - sp_count)
            if idx + '_' + wrd in new_w2p:
                # 是 new_str phn 序列的 index
                left_idx += len(new_w2p[idx + '_' + wrd])
                # old phn 序列
                new_phns_left.extend(w2p[key])
            else:
                span_to_repl[0] = len(new_phns_left)
                span_to_add[0] = len(new_phns_left)
                break

    # reverse w2p and new_w2p
    right_idx = 0
    new_phns_right = []
    sp_count = 0
    w2p_max_idx = _get_max_idx(w2p)
    new_w2p_max_idx = _get_max_idx(new_w2p)
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
        for key in list(w2p.keys())[::-1]:
            idx, wrd = key.split('_')
            if wrd == 'sp':
                sp_count += 1
                new_phns_right = ['sp'] + new_phns_right
            else:
                idx = str(new_w2p_max_idx - (w2p_max_idx - int(idx) - sp_count))
                if idx + '_' + wrd in new_w2p:
                    right_idx -= len(new_w2p[idx + '_' + wrd])
                    new_phns_right = w2p[key] + new_phns_right
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
    outs = {}
    outs['mfa_start'] = mfa_start
    outs['mfa_end'] = mfa_end
    outs['old_phns'] = old_phns
    outs['new_phns'] = new_phns
    outs['span_to_repl'] = span_to_repl
    outs['span_to_add'] = span_to_add

    return outs


if __name__ == '__main__':
    text = "For that reason cover should not be given."
    phn, dur, word2phns = alignment("exp/p243_313.wav", text, lang='en')
    print(phn, dur)
    print(word2phns)
    print("---------------------------------")
    # 这里可以用我们的中文前端得到 pinyin 序列
    text_zh = "卡尔普陪外孙玩滑梯。"
    text_zh = pypinyin.lazy_pinyin(
        text_zh,
        neutral_tone_with_five=True,
        style=pypinyin.Style.TONE3,
        tone_sandhi=True)
    text_zh = " ".join(text_zh)
    phn, dur, word2phns = alignment("exp/000001.wav", text_zh, lang='zh')
    print(phn, dur)
    print(word2phns)
    print("---------------------------------")
    phns, wrd2phns = words2phns(text, lang='en')
    print("phns:", phns)
    print("wrd2phns:", wrd2phns)
    print("---------------------------------")

    phns, wrd2phns = words2phns(text_zh, lang='zh')
    print("phns:", phns)
    print("wrd2phns:", wrd2phns)
    print("---------------------------------")

    outs = get_phns_spans(
        wav_path="exp/p243_313.wav",
        old_str="For that reason cover should not be given.",
        new_str="for that reason cover is impossible to be given.")

    mfa_start = outs["mfa_start"]
    mfa_end = outs["mfa_end"]
    old_phns = outs["old_phns"]
    new_phns = outs["new_phns"]
    span_to_repl = outs["span_to_repl"]
    span_to_add = outs["span_to_add"]
    print("mfa_start:", mfa_start)
    print("mfa_end:", mfa_end)
    print("old_phns:", old_phns)
    print("new_phns:", new_phns)
    print("span_to_repl:", span_to_repl)
    print("span_to_add:", span_to_add)
    print("---------------------------------")
