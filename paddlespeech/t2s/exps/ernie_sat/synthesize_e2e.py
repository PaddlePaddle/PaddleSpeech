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
import librosa
import numpy as np
import soundfile as sf

from paddlespeech.t2s.datasets.am_batch_fn import build_erniesat_collate_fn
from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.exps.ernie_sat.align import get_phns_spans
from paddlespeech.t2s.exps.ernie_sat.utils import eval_durs
from paddlespeech.t2s.exps.ernie_sat.utils import get_dur_adj_factor
from paddlespeech.t2s.exps.ernie_sat.utils import get_span_bdy
from paddlespeech.t2s.exps.ernie_sat.utils import get_tmp_name
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import norm


def _p2id(self, phonemes: List[str]) -> np.ndarray:
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
                        n_shift: int=300):
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
        n_shift=n_shift)

    mfa_start = phns_spans_outs["mfa_start"]
    mfa_end = phns_spans_outs["mfa_end"]
    old_phns = phns_spans_outs["old_phns"]
    new_phns = phns_spans_outs["new_phns"]
    span_to_repl = phns_spans_outs["span_to_repl"]
    span_to_add = phns_spans_outs["span_to_add"]

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

    # 音频是正常遮住了
    sf.write(str("new_wav.wav"), new_wav, samplerate=fs)

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
               old_str: str='',
               new_str: str='',
               source_lang: str='en',
               target_lang: str='en',
               duration_adjust: bool=True,
               fs: int=24000,
               n_shift: int=300):

    outs = prep_feats_with_dur(
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        source_lang=source_lang,
        target_lang=target_lang,
        duration_adjust=duration_adjust,
        fs=fs,
        n_shift=n_shift)

    wav_name = os.path.basename(wav_path)
    utt_id = wav_name.split('.')[0]

    wav = outs['new_wav']
    phns = outs['new_phns']
    mfa_start = outs['new_mfa_start']
    mfa_end = outs['new_mfa_end']
    old_span_bdy = outs['old_span_bdy']
    new_span_bdy = outs['new_span_bdy']
    span_bdy = np.array(new_span_bdy)

    text = _p2id(phns)
    mel = mel_extractor.get_log_mel_fbank(wav)
    erniesat_mean, erniesat_std = np.load(erniesat_stat)
    normed_mel = norm(mel, erniesat_mean, erniesat_std)
    tmp_name = get_tmp_name(text=old_str)
    tmpbase = './tmp_dir/' + tmp_name
    tmpbase = Path(tmpbase)
    tmpbase.mkdir(parents=True, exist_ok=True)
    print("tmp_name in synthesize_e2e:", tmp_name)

    mel_path = tmpbase / 'mel.npy'
    print("mel_path:", mel_path)
    np.save(mel_path, logmel)
    durations = [e - s for e, s in zip(mfa_end, mfa_start)]

    datum = {
        "utt_id": utt_id,
        "spk_id": 0,
        "text": text,
        "text_lengths": len(text),
        "speech_lengths": 115,
        "durations": durations,
        "speech": mel_path,
        "align_start": mfa_start,
        "align_end": mfa_end,
        "span_bdy": span_bdy
    }

    batch = collate_fn([datum])
    print("batch:", batch)

    return batch, old_span_bdy, new_span_bdy


def decode_with_model(mlm_model: nn.Layer,
                      collate_fn,
                      wav_path: str,
                      old_str: str='',
                      new_str: str='',
                      source_lang: str='en',
                      target_lang: str='en',
                      use_teacher_forcing: bool=False,
                      duration_adjust: bool=True,
                      fs: int=24000,
                      n_shift: int=300,
                      token_list: List[str]=[]):
    batch, old_span_bdy, new_span_bdy = prep_feats(
        source_lang=source_lang,
        target_lang=target_lang,
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        duration_adjust=duration_adjust,
        fs=fs,
        n_shift=n_shift,
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


if __name__ == '__main__':
    fs = 24000
    n_shift = 300
    wav_path = "exp/p243_313.wav"
    old_str = "For that reason cover should not be given."
    # for edit
    # new_str = "for that reason cover is impossible to be given."
    # for synthesize
    append_str = "do you love me i love you so much"
    new_str = old_str + append_str
    '''
    outs = prep_feats_with_dur(
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        fs=fs,
        n_shift=n_shift)

    new_wav = outs['new_wav']
    new_phns = outs['new_phns']
    new_mfa_start = outs['new_mfa_start']
    new_mfa_end = outs['new_mfa_end']
    old_span_bdy = outs['old_span_bdy']
    new_span_bdy = outs['new_span_bdy']

    print("---------------------------------")

    print("new_wav:", new_wav)
    print("new_phns:", new_phns)
    print("new_mfa_start:", new_mfa_start)
    print("new_mfa_end:", new_mfa_end)
    print("old_span_bdy:", old_span_bdy)
    print("new_span_bdy:", new_span_bdy)
    print("---------------------------------")
    '''

    erniesat_config = "/home/yuantian01/PaddleSpeech_ERNIE_SAT/PaddleSpeech/examples/vctk/ernie_sat/local/default.yaml"

    with open(erniesat_config) as f:
        erniesat_config = CfgNode(yaml.safe_load(f))

    erniesat_stat = "/home/yuantian01/PaddleSpeech_ERNIE_SAT/PaddleSpeech/examples/vctk/ernie_sat/dump/train/speech_stats.npy"

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

    phones_dict = '/home/yuantian01/PaddleSpeech_ERNIE_SAT/PaddleSpeech/examples/vctk/ernie_sat/dump/phone_id_map.txt'
    vocab_phones = {}

    with open(phones_dict, 'rt') as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    for phn, id in phn_id:
        vocab_phones[phn] = int(id)

    prep_feats(
        wav_path=wav_path,
        old_str=old_str,
        new_str=new_str,
        fs=fs,
        n_shift=n_shift)
