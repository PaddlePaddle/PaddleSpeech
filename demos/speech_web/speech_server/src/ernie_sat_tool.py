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
import argparse
import os
from pathlib import Path
from typing import List
from unittest import main

import librosa
import numpy as np
import paddle
import pypinyin
import soundfile as sf
import yaml
from pypinyin_dict.phrase_pinyin_data import large_pinyin
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.am_batch_fn import build_erniesat_collate_fn
from paddlespeech.t2s.datasets.get_feats import LogMelFBank
# from paddlespeech.t2s.exps.ernie_sat.align import get_phns_spans
from paddlespeech.t2s.exps.ernie_sat.utils import get_dur_adj_factor
from paddlespeech.t2s.exps.ernie_sat.utils import get_span_bdy
from paddlespeech.t2s.exps.ernie_sat.utils import get_tmp_name
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.exps.syn_utils import norm
from paddlespeech.t2s.utils import str2bool
large_pinyin.load()

from .align import get_phns_spans

def eval_durs(phns, target_lang: str='zh', fs: int=24000, n_shift: int=300):
    
    if target_lang == 'en':
        am = "fastspeech2_ljspeech"
        am_config = "source/model/fastspeech2_nosil_ljspeech_ckpt_0.5/default.yaml"
        am_ckpt = "source/model/fastspeech2_nosil_ljspeech_ckpt_0.5/snapshot_iter_100000.pdz"
        am_stat = "source/model/fastspeech2_nosil_ljspeech_ckpt_0.5/speech_stats.npy"
        phones_dict = "source/model/fastspeech2_nosil_ljspeech_ckpt_0.5/phone_id_map.txt"

    elif target_lang == 'zh':
        am = "fastspeech2_csmsc"
        am_config = "source/model/fastspeech2_conformer_baker_ckpt_0.5/conformer.yaml"
        am_ckpt = "source/model/fastspeech2_conformer_baker_ckpt_0.5/snapshot_iter_76000.pdz"
        am_stat = "source/model/fastspeech2_conformer_baker_ckpt_0.5/speech_stats.npy"
        phones_dict = "source/model/fastspeech2_conformer_baker_ckpt_0.5/phone_id_map.txt"

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



def _p2id(phonemes: List[str], vocab_phones) -> np.ndarray:
    # replace unk phone with sp
    phonemes = [phn if phn in vocab_phones else "sp" for phn in phonemes]
    phone_ids = [vocab_phones[item] for item in phonemes]
    return np.array(phone_ids, np.int64)


def prep_feats_with_dur(wav_path: str,
                        old_str: str='',
                        new_str: str='',
                        source_lang: str='en',
                        target_lang: str='en',
                        duration_adjust: bool=True,
                        fs: int=24000,
                        n_shift: int=300,
                        mfa_version='v1'):
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
    phns_spans_outs = get_phns_spans(
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        source_lang=source_lang,
        target_lang=target_lang,
        fs=fs,
        n_shift=n_shift,
        mfa_version=mfa_version)

    mfa_start = phns_spans_outs['mfa_start']
    mfa_end = phns_spans_outs['mfa_end']
    old_phns = phns_spans_outs['old_phns']
    new_phns = phns_spans_outs['new_phns']
    span_to_repl = phns_spans_outs['span_to_repl']
    span_to_add = phns_spans_outs['span_to_add']

    # 中文的 phns 不一定都在 fastspeech2 的字典里, 用 sp 代替
    if target_lang in {'en', 'zh'}:
        old_durs = eval_durs(old_phns, target_lang=source_lang)
    else:
        assert target_lang in {'en', 'zh'}, \
            "calculate duration_predict is not support for this language..."

    orig_old_durs = [e - s for e, s in zip(mfa_end, mfa_start)]

    if duration_adjust:
        d_factor = get_dur_adj_factor(
            orig_dur=orig_old_durs, pred_dur=old_durs, phns=old_phns)
        d_factor = d_factor * 1.25
    else:
        d_factor = 1

    if target_lang in {'en', 'zh'}:
        new_durs = eval_durs(new_phns, target_lang=target_lang)
    else:
        assert target_lang == "zh" or target_lang == "en", \
            "calculate duration_predict is not support for this language..."

    # duration 要是整数
    new_durs_adjusted = [int(np.ceil(d_factor * i)) for i in new_durs]

    new_span_dur_sum = sum(new_durs_adjusted[span_to_add[0]:span_to_add[1]])
    old_span_dur_sum = sum(orig_old_durs[span_to_repl[0]:span_to_repl[1]])
    dur_offset = new_span_dur_sum - old_span_dur_sum
    new_mfa_start = mfa_start[:span_to_repl[0]]
    new_mfa_end = mfa_end[:span_to_repl[0]]

    for dur in new_durs_adjusted[span_to_add[0]:span_to_add[1]]:
        if len(new_mfa_end) == 0:
            new_mfa_start.append(0)
            new_mfa_end.append(dur)
        else:
            new_mfa_start.append(new_mfa_end[-1])
            new_mfa_end.append(new_mfa_end[-1] + dur)

    new_mfa_start += [i + dur_offset for i in mfa_start[span_to_repl[1]:]]
    new_mfa_end += [i + dur_offset for i in mfa_end[span_to_repl[1]:]]

    # 3. get new wav
    # 在原始句子后拼接
    if span_to_repl[0] >= len(mfa_start):
        wav_left_idx = len(wav_org)
        wav_right_idx = wav_left_idx
    # 在原始句子中间替换
    else:
        wav_left_idx = int(np.floor(mfa_start[span_to_repl[0]] * n_shift))
        wav_right_idx = int(np.ceil(mfa_end[span_to_repl[1] - 1] * n_shift))
    blank_wav = np.zeros(
        (int(np.ceil(new_span_dur_sum * n_shift)), ), dtype=wav_org.dtype)
    # 原始音频，需要编辑的部分替换成空音频，空音频的时间由 fs2 的 duration_predictor 决定
    new_wav = np.concatenate(
        [wav_org[:wav_left_idx], blank_wav, wav_org[wav_right_idx:]])

    # 4. get old and new mel span to be mask
    old_span_bdy = get_span_bdy(
        mfa_start=mfa_start, mfa_end=mfa_end, span_to_repl=span_to_repl)

    new_span_bdy = get_span_bdy(
        mfa_start=new_mfa_start, mfa_end=new_mfa_end, span_to_repl=span_to_add)

    # old_span_bdy, new_span_bdy 是帧级别的范围
    outs = {}
    outs['new_wav'] = new_wav
    outs['new_phns'] = new_phns
    outs['new_mfa_start'] = new_mfa_start
    outs['new_mfa_end'] = new_mfa_end
    outs['old_span_bdy'] = old_span_bdy
    outs['new_span_bdy'] = new_span_bdy
    return outs


def prep_feats(wav_path: str,
               mel_extractor,
               vocab_phones,
               erniesat_stat,
               collate_fn,
               old_str: str='',
               new_str: str='',
               source_lang: str='en',
               target_lang: str='en',
               duration_adjust: bool=True,
               fs: int=24000,
               n_shift: int=300,
               mfa_version: str='v1'
               ):

    with_dur_outs = prep_feats_with_dur(
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        source_lang=source_lang,
        target_lang=target_lang,
        duration_adjust=duration_adjust,
        fs=fs,
        n_shift=n_shift,
        mfa_version=mfa_version
        )

    wav_name = os.path.basename(wav_path)
    utt_id = wav_name.split('.')[0]

    wav = with_dur_outs['new_wav']
    phns = with_dur_outs['new_phns']
    mfa_start = with_dur_outs['new_mfa_start']
    mfa_end = with_dur_outs['new_mfa_end']
    old_span_bdy = with_dur_outs['old_span_bdy']
    new_span_bdy = with_dur_outs['new_span_bdy']
    span_bdy = np.array(new_span_bdy)

    mel = mel_extractor.get_log_mel_fbank(wav)
    erniesat_mean, erniesat_std = np.load(erniesat_stat)
    normed_mel = norm(mel, erniesat_mean, erniesat_std)
    tmp_name = 'ernie_sat/' + get_tmp_name(text=old_str)
    tmpbase = './tmp_dir/' + tmp_name
    tmpbase = Path(tmpbase)
    tmpbase.mkdir(parents=True, exist_ok=True)

    mel_path = tmpbase / 'mel.npy'
    np.save(mel_path, normed_mel)
    durations = [e - s for e, s in zip(mfa_end, mfa_start)]
    text = _p2id(phns, vocab_phones)

    datum = {
        "utt_id": utt_id,
        "spk_id": 0,
        "text": text,
        "text_lengths": len(text),
        "speech_lengths": len(normed_mel),
        "durations": durations,
        "speech": np.load(mel_path),
        "align_start": mfa_start,
        "align_end": mfa_end,
        "span_bdy": span_bdy
    }

    batch = collate_fn([datum])
    outs = dict()
    outs['batch'] = batch
    outs['old_span_bdy'] = old_span_bdy
    outs['new_span_bdy'] = new_span_bdy
    return outs


def get_mlm_output(wav_path: str,
                   erniesat_inference,
                   mel_extractor,
                    vocab_phones,
                    erniesat_stat,
                    collate_fn,
                   old_str: str='',
                   new_str: str='',
                   source_lang: str='en',
                   target_lang: str='en',
                   duration_adjust: bool=True,
                   fs: int=24000,
                   n_shift: int=300,
                   mfa_version: str='v1' ):

    prep_feats_outs = prep_feats(
        wav_path=wav_path,
        mel_extractor=mel_extractor,
        vocab_phones=vocab_phones,
        erniesat_stat=erniesat_stat,
        collate_fn=collate_fn,
        old_str=old_str,
        new_str=new_str,
        source_lang=source_lang,
        target_lang=target_lang,
        duration_adjust=duration_adjust,
        fs=fs,
        n_shift=n_shift,
        mfa_version=mfa_version)

    batch = prep_feats_outs['batch']
    new_span_bdy = prep_feats_outs['new_span_bdy']
    old_span_bdy = prep_feats_outs['old_span_bdy']

    out_mels = erniesat_inference(
        speech=batch['speech'],
        text=batch['text'],
        masked_pos=batch['masked_pos'],
        speech_mask=batch['speech_mask'],
        text_mask=batch['text_mask'],
        speech_seg_pos=batch['speech_seg_pos'],
        text_seg_pos=batch['text_seg_pos'],
        span_bdy=new_span_bdy)

    # 拼接音频
    output_feat = paddle.concat(x=out_mels, axis=0)
    wav_org, _ = librosa.load(wav_path, sr=fs)
    outs = dict()
    outs['wav_org'] = wav_org
    outs['output_feat'] = output_feat
    outs['old_span_bdy'] = old_span_bdy
    outs['new_span_bdy'] = new_span_bdy

    return outs


def get_wav(wav_path: str,
            task_name,
            voc_inference,
            erniesat_inference,
            mel_extractor,
            vocab_phones,
            erniesat_stat,
            collate_fn,
            source_lang: str='en',
            target_lang: str='en',
            old_str: str='',
            new_str: str='',
            duration_adjust: bool=True,
            fs: int=24000,
            n_shift: int=300,
            mfa_version: str='v1'):

    outs = get_mlm_output(
        wav_path=wav_path,
        erniesat_inference=erniesat_inference,
        mel_extractor=mel_extractor,
        vocab_phones=vocab_phones,
        erniesat_stat=erniesat_stat,
        collate_fn=collate_fn,
        old_str=old_str,
        new_str=new_str,
        source_lang=source_lang,
        target_lang=target_lang,
        duration_adjust=duration_adjust,
        fs=fs,
        n_shift=n_shift,
        mfa_version=mfa_version)

    wav_org = outs['wav_org']
    output_feat = outs['output_feat']
    old_span_bdy = outs['old_span_bdy']
    new_span_bdy = outs['new_span_bdy']

    masked_feat = output_feat[new_span_bdy[0]:new_span_bdy[1]]

    with paddle.no_grad():
        alt_wav = voc_inference(masked_feat)
    alt_wav = np.squeeze(alt_wav)

    old_time_bdy = [n_shift * x for x in old_span_bdy]
    if task_name == 'edit':
        wav_replaced = np.concatenate(
            [wav_org[:old_time_bdy[0]], alt_wav, wav_org[old_time_bdy[1]:]])
    else:
        wav_replaced = alt_wav

    wav_dict = {"origin": wav_org, "output": wav_replaced}
    return wav_dict


def ernie_sat_web(erniesat_config,
                  old_str,
                  new_str,
                  source_lang,
                  target_lang, 
                  task_name,
                  erniesat_ckpt,
                  erniesat_stat,
                  phones_dict,
                  voc_config,
                  voc,
                  voc_ckpt,
                  voc_stat,
                  duration_adjust,
                  wav_path,
                  output_name,
                  mfa_version='v1'               
                  ):
    with open(erniesat_config) as f:
        erniesat_config = CfgNode(yaml.safe_load(f))

    # convert Chinese characters to pinyin
    if source_lang == 'zh':
        old_str = pypinyin.lazy_pinyin(
            old_str,
            neutral_tone_with_five=True,
            style=pypinyin.Style.TONE3,
            tone_sandhi=True)
        old_str = ' '.join(old_str)
    if target_lang == 'zh':
        new_str = pypinyin.lazy_pinyin(
            new_str,
            neutral_tone_with_five=True,
            style=pypinyin.Style.TONE3,
            tone_sandhi=True)
        new_str = ' '.join(new_str)

    if task_name == 'edit':
        new_str = new_str
    elif task_name == 'synthesize':
        new_str = old_str + ' ' + new_str
    else:
        new_str = old_str + ' ' + new_str
    print("new_str:", new_str)

    # Extractor
    mel_extractor = LogMelFBank(
        sr=erniesat_config.fs,
        n_fft=erniesat_config.n_fft,
        hop_length=erniesat_config.n_shift,
        win_length=erniesat_config.win_length,
        window=erniesat_config.window,
        n_mels=erniesat_config.n_mels,
        fmin=erniesat_config.fmin,
        fmax=erniesat_config.fmax)

    collate_fn = build_erniesat_collate_fn(
        mlm_prob=erniesat_config.mlm_prob,
        mean_phn_span=erniesat_config.mean_phn_span,
        seg_emb=erniesat_config.model['enc_input_layer'] == 'sega_mlm',
        text_masking=False)

    vocab_phones = {}

    with open(phones_dict, 'rt') as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    for phn, id in phn_id:
        vocab_phones[phn] = int(id)

    # ernie sat model
    erniesat_inference = get_am_inference(
        am='erniesat_dataset',
        am_config=erniesat_config,
        am_ckpt=erniesat_ckpt,
        am_stat=erniesat_stat,
        phones_dict=phones_dict)

    with open(voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))

    # vocoder
    voc_inference = get_voc_inference(
        voc=voc,
        voc_config=voc_config,
        voc_ckpt=voc_ckpt,
        voc_stat=voc_stat)

    erniesat_stat = erniesat_stat

    wav_dict = get_wav(
        wav_path=wav_path,
        task_name=task_name,
        voc_inference=voc_inference,
        erniesat_inference=erniesat_inference,
        mel_extractor=mel_extractor,
        vocab_phones=vocab_phones,
        erniesat_stat=erniesat_stat,
        collate_fn=collate_fn,
        source_lang=source_lang,
        target_lang=target_lang,
        old_str=old_str,
        new_str=new_str,
        duration_adjust=duration_adjust,
        fs=erniesat_config.fs,
        n_shift=erniesat_config.n_shift,
        mfa_version=mfa_version)

    sf.write(
        output_name, wav_dict['output'], samplerate=erniesat_config.fs)
    return output_name


if __name__ == '__main__':
    
    erniesat_config = "source/model/erniesat_aishell3_ckpt_1.2.0/default.yaml"
    erniesat_ckpt = "source/model/erniesat_aishell3_ckpt_1.2.0/snapshot_iter_289500.pdz"
    erniesat_stat = "source/model/erniesat_aishell3_ckpt_1.2.0/speech_stats.npy"
    phones_dict = "source/model/erniesat_aishell3_ckpt_1.2.0/phone_id_map.txt"
    duration_adjust = True
    
    voc = "hifigan_aishell3"
    voc_config = "source/model/hifigan_aishell3_ckpt_0.2.0/default.yaml"
    voc_ckpt = "source/model/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz"
    voc_stat = "source/model/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy"
    
    
    old_str = "今天天气很好"
    new_str = "今天心情很好"
    source_lang = "zh"
    target_lang = "zh"
    task_name = "edit"
    wav_path = "source/wav/SAT/upload/SSB03540428.wav"
    output_name = "source/wav/SAT/out/demo_edit.wav"
    
    mfa_version='v2'
    
    ernie_sat_web(erniesat_config,
                  old_str,
                  new_str,
                  source_lang,
                  target_lang, 
                  task_name,
                  erniesat_ckpt,
                  erniesat_stat,
                  phones_dict,
                  voc_config,
                  voc,
                  voc_ckpt,
                  voc_stat,
                  duration_adjust,
                  wav_path,
                  output_name,
                  mfa_version=mfa_version               
                  )
    