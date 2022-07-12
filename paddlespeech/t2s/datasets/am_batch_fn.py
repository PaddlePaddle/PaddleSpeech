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
from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import paddle

from paddlespeech.t2s.datasets.batch import batch_sequences
from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.modules.nets_utils import get_seg_pos
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask
from paddlespeech.t2s.modules.nets_utils import pad_list
from paddlespeech.t2s.modules.nets_utils import phones_masking
from paddlespeech.t2s.modules.nets_utils import phones_text_masking


# 因为要传参数，所以需要额外构建
def build_erniesat_collate_fn(
        mlm_prob: float=0.8,
        mean_phn_span: int=8,
        seg_emb: bool=False,
        text_masking: bool=False,
        epoch: int=-1, ):

    if epoch == -1:
        mlm_prob_factor = 1
    else:
        mlm_prob_factor = 0.8

    return ErnieSATCollateFn(
        mlm_prob=mlm_prob * mlm_prob_factor,
        mean_phn_span=mean_phn_span,
        seg_emb=seg_emb,
        text_masking=text_masking)


class ErnieSATCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(self,
                 mlm_prob: float=0.8,
                 mean_phn_span: int=8,
                 seg_emb: bool=False,
                 text_masking: bool=False):
        self.mlm_prob = mlm_prob
        self.mean_phn_span = mean_phn_span
        self.seg_emb = seg_emb
        self.text_masking = text_masking

    def __call__(self, exmaples):
        return erniesat_batch_fn(
            exmaples,
            mlm_prob=self.mlm_prob,
            mean_phn_span=self.mean_phn_span,
            seg_emb=self.seg_emb,
            text_masking=self.text_masking)


def erniesat_batch_fn(examples,
                      mlm_prob: float=0.8,
                      mean_phn_span: int=8,
                      seg_emb: bool=False,
                      text_masking: bool=False):
    # fields = ["text", "text_lengths", "speech", "speech_lengths", "durations", "pitch", "energy"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]

    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    align_start = [
        np.array(item["align_start"], dtype=np.int64) for item in examples
    ]

    align_end = [
        np.array(item["align_end"], dtype=np.int64) for item in examples
    ]

    align_start_lengths = [
        np.array(len(item["align_start"]), dtype=np.int64) for item in examples
    ]

    # add_pad
    text = batch_sequences(text)
    speech = batch_sequences(speech)
    align_start = batch_sequences(align_start)
    align_end = batch_sequences(align_end)

    # convert each batch to paddle.Tensor
    text = paddle.to_tensor(text)
    speech = paddle.to_tensor(speech)
    text_lengths = paddle.to_tensor(text_lengths)
    speech_lengths = paddle.to_tensor(speech_lengths)
    align_start_lengths = paddle.to_tensor(align_start_lengths)

    speech_pad = speech
    text_pad = text

    text_mask = make_non_pad_mask(
        text_lengths, text_pad, length_dim=1).unsqueeze(-2)
    speech_mask = make_non_pad_mask(
        speech_lengths, speech_pad[:, :, 0], length_dim=1).unsqueeze(-2)

    # for training
    span_bdy = None
    # for inference
    if 'span_bdy' in examples[0].keys():
        span_bdy = [
            np.array(item["span_bdy"], dtype=np.int64) for item in examples
        ]
        span_bdy = paddle.to_tensor(span_bdy)

    # dual_mask 的是混合中英时候同时 mask 语音和文本 
    # ernie sat 在实现跨语言的时候都 mask 了
    if text_masking:
        masked_pos, text_masked_pos = phones_text_masking(
            xs_pad=speech_pad,
            src_mask=speech_mask,
            text_pad=text_pad,
            text_mask=text_mask,
            align_start=align_start,
            align_end=align_end,
            align_start_lens=align_start_lengths,
            mlm_prob=mlm_prob,
            mean_phn_span=mean_phn_span,
            span_bdy=span_bdy)
    # 训练纯中文和纯英文的 -> a3t 没有对 phoneme 做 mask, 只对语音 mask 了
    # a3t 和 ernie sat 的区别主要在于做 mask 的时候
    else:
        masked_pos = phones_masking(
            xs_pad=speech_pad,
            src_mask=speech_mask,
            align_start=align_start,
            align_end=align_end,
            align_start_lens=align_start_lengths,
            mlm_prob=mlm_prob,
            mean_phn_span=mean_phn_span,
            span_bdy=span_bdy)
        text_masked_pos = paddle.zeros(paddle.shape(text_pad))

    speech_seg_pos, text_seg_pos = get_seg_pos(
        speech_pad=speech_pad,
        text_pad=text_pad,
        align_start=align_start,
        align_end=align_end,
        align_start_lens=align_start_lengths,
        seg_emb=seg_emb)

    batch = {
        "text": text,
        "speech": speech,
        # need to generate 
        "masked_pos": masked_pos,
        "speech_mask": speech_mask,
        "text_mask": text_mask,
        "speech_seg_pos": speech_seg_pos,
        "text_seg_pos": text_seg_pos,
        "text_masked_pos": text_masked_pos
    }

    return batch


def tacotron2_single_spk_batch_fn(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    speech = batch_sequences(speech)

    # convert each batch to paddle.Tensor
    text = paddle.to_tensor(text)
    speech = paddle.to_tensor(speech)
    text_lengths = paddle.to_tensor(text_lengths)
    speech_lengths = paddle.to_tensor(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "speech": speech,
        "speech_lengths": speech_lengths,
    }
    return batch


def tacotron2_multi_spk_batch_fn(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    speech = batch_sequences(speech)

    # convert each batch to paddle.Tensor
    text = paddle.to_tensor(text)
    speech = paddle.to_tensor(speech)
    text_lengths = paddle.to_tensor(text_lengths)
    speech_lengths = paddle.to_tensor(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "speech": speech,
        "speech_lengths": speech_lengths,
    }
    # spk_emb has a higher priority than spk_id
    if "spk_emb" in examples[0]:
        spk_emb = [
            np.array(item["spk_emb"], dtype=np.float32) for item in examples
        ]
        spk_emb = batch_sequences(spk_emb)
        spk_emb = paddle.to_tensor(spk_emb)
        batch["spk_emb"] = spk_emb
    elif "spk_id" in examples[0]:
        spk_id = [np.array(item["spk_id"], dtype=np.int64) for item in examples]
        spk_id = paddle.to_tensor(spk_id)
        batch["spk_id"] = spk_id
    return batch


def speedyspeech_single_spk_batch_fn(examples):
    # fields = ["phones", "tones", "num_phones", "num_frames", "feats", "durations"]
    phones = [np.array(item["phones"], dtype=np.int64) for item in examples]
    tones = [np.array(item["tones"], dtype=np.int64) for item in examples]
    feats = [np.array(item["feats"], dtype=np.float32) for item in examples]
    durations = [
        np.array(item["durations"], dtype=np.int64) for item in examples
    ]
    num_phones = [
        np.array(item["num_phones"], dtype=np.int64) for item in examples
    ]
    num_frames = [
        np.array(item["num_frames"], dtype=np.int64) for item in examples
    ]

    phones = batch_sequences(phones)
    tones = batch_sequences(tones)
    feats = batch_sequences(feats)
    durations = batch_sequences(durations)

    # convert each batch to paddle.Tensor
    phones = paddle.to_tensor(phones)
    tones = paddle.to_tensor(tones)
    feats = paddle.to_tensor(feats)
    durations = paddle.to_tensor(durations)
    num_phones = paddle.to_tensor(num_phones)
    num_frames = paddle.to_tensor(num_frames)
    batch = {
        "phones": phones,
        "tones": tones,
        "num_phones": num_phones,
        "num_frames": num_frames,
        "feats": feats,
        "durations": durations,
    }
    return batch


def speedyspeech_multi_spk_batch_fn(examples):
    # fields = ["phones", "tones", "num_phones", "num_frames", "feats", "durations", "spk_id"]
    phones = [np.array(item["phones"], dtype=np.int64) for item in examples]
    tones = [np.array(item["tones"], dtype=np.int64) for item in examples]
    feats = [np.array(item["feats"], dtype=np.float32) for item in examples]
    durations = [
        np.array(item["durations"], dtype=np.int64) for item in examples
    ]
    num_phones = [
        np.array(item["num_phones"], dtype=np.int64) for item in examples
    ]
    num_frames = [
        np.array(item["num_frames"], dtype=np.int64) for item in examples
    ]

    phones = batch_sequences(phones)
    tones = batch_sequences(tones)
    feats = batch_sequences(feats)
    durations = batch_sequences(durations)

    # convert each batch to paddle.Tensor
    phones = paddle.to_tensor(phones)
    tones = paddle.to_tensor(tones)
    feats = paddle.to_tensor(feats)
    durations = paddle.to_tensor(durations)
    num_phones = paddle.to_tensor(num_phones)
    num_frames = paddle.to_tensor(num_frames)
    batch = {
        "phones": phones,
        "tones": tones,
        "num_phones": num_phones,
        "num_frames": num_frames,
        "feats": feats,
        "durations": durations,
    }
    if "spk_id" in examples[0]:
        spk_id = [np.array(item["spk_id"], dtype=np.int64) for item in examples]
        spk_id = paddle.to_tensor(spk_id)
        batch["spk_id"] = spk_id
    return batch


def fastspeech2_single_spk_batch_fn(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths", "durations", "pitch", "energy"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    pitch = [np.array(item["pitch"], dtype=np.float32) for item in examples]
    energy = [np.array(item["energy"], dtype=np.float32) for item in examples]
    durations = [
        np.array(item["durations"], dtype=np.int64) for item in examples
    ]

    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    pitch = batch_sequences(pitch)
    speech = batch_sequences(speech)
    durations = batch_sequences(durations)
    energy = batch_sequences(energy)

    # convert each batch to paddle.Tensor
    text = paddle.to_tensor(text)
    pitch = paddle.to_tensor(pitch)
    speech = paddle.to_tensor(speech)
    durations = paddle.to_tensor(durations)
    energy = paddle.to_tensor(energy)
    text_lengths = paddle.to_tensor(text_lengths)
    speech_lengths = paddle.to_tensor(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "durations": durations,
        "speech": speech,
        "speech_lengths": speech_lengths,
        "pitch": pitch,
        "energy": energy
    }
    return batch


def fastspeech2_multi_spk_batch_fn(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths", "durations", "pitch", "energy", "spk_id"/"spk_emb"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    pitch = [np.array(item["pitch"], dtype=np.float32) for item in examples]
    energy = [np.array(item["energy"], dtype=np.float32) for item in examples]
    durations = [
        np.array(item["durations"], dtype=np.int64) for item in examples
    ]
    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    pitch = batch_sequences(pitch)
    speech = batch_sequences(speech)
    durations = batch_sequences(durations)
    energy = batch_sequences(energy)

    # convert each batch to paddle.Tensor
    text = paddle.to_tensor(text)
    pitch = paddle.to_tensor(pitch)
    speech = paddle.to_tensor(speech)
    durations = paddle.to_tensor(durations)
    energy = paddle.to_tensor(energy)
    text_lengths = paddle.to_tensor(text_lengths)
    speech_lengths = paddle.to_tensor(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "durations": durations,
        "speech": speech,
        "speech_lengths": speech_lengths,
        "pitch": pitch,
        "energy": energy
    }
    # spk_emb has a higher priority than spk_id
    if "spk_emb" in examples[0]:
        spk_emb = [
            np.array(item["spk_emb"], dtype=np.float32) for item in examples
        ]
        spk_emb = batch_sequences(spk_emb)
        spk_emb = paddle.to_tensor(spk_emb)
        batch["spk_emb"] = spk_emb
    elif "spk_id" in examples[0]:
        spk_id = [np.array(item["spk_id"], dtype=np.int64) for item in examples]
        spk_id = paddle.to_tensor(spk_id)
        batch["spk_id"] = spk_id
    return batch


def transformer_single_spk_batch_fn(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    speech = batch_sequences(speech)

    # convert each batch to paddle.Tensor
    text = paddle.to_tensor(text)
    speech = paddle.to_tensor(speech)
    text_lengths = paddle.to_tensor(text_lengths)
    speech_lengths = paddle.to_tensor(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "speech": speech,
        "speech_lengths": speech_lengths,
    }
    return batch


def vits_single_spk_batch_fn(examples):
    """
    Returns:
        Dict[str, Any]:
            - text (Tensor): Text index tensor (B, T_text).
            - text_lengths (Tensor): Text length tensor (B,).
            - feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            - feats_lengths (Tensor): Feature length tensor (B,).
            - speech (Tensor): Speech waveform tensor (B, T_wav).

    """
    # fields = ["text", "text_lengths", "feats", "feats_lengths", "speech"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    feats = [np.array(item["feats"], dtype=np.float32) for item in examples]
    speech = [np.array(item["wave"], dtype=np.float32) for item in examples]
    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    feats_lengths = [
        np.array(item["feats_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    feats = batch_sequences(feats)
    speech = batch_sequences(speech)

    # convert each batch to paddle.Tensor
    text = paddle.to_tensor(text)
    feats = paddle.to_tensor(feats)
    text_lengths = paddle.to_tensor(text_lengths)
    feats_lengths = paddle.to_tensor(feats_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "feats": feats,
        "feats_lengths": feats_lengths,
        "speech": speech
    }
    return batch


# for ERNIE SAT
class MLMCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
            self,
            feats_extract,
            mlm_prob: float=0.8,
            mean_phn_span: int=8,
            seg_emb: bool=False,
            text_masking: bool=False,
            attention_window: int=0,
            not_sequence: Collection[str]=(), ):
        self.mlm_prob = mlm_prob
        self.mean_phn_span = mean_phn_span
        self.feats_extract = feats_extract
        self.not_sequence = set(not_sequence)
        self.attention_window = attention_window
        self.seg_emb = seg_emb
        self.text_masking = text_masking

    def __call__(self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
                 ) -> Tuple[List[str], Dict[str, paddle.Tensor]]:
        return mlm_collate_fn(
            data,
            feats_extract=self.feats_extract,
            mlm_prob=self.mlm_prob,
            mean_phn_span=self.mean_phn_span,
            seg_emb=self.seg_emb,
            text_masking=self.text_masking,
            not_sequence=self.not_sequence)


def mlm_collate_fn(
        data: Collection[Tuple[str, Dict[str, np.ndarray]]],
        feats_extract=None,
        mlm_prob: float=0.8,
        mean_phn_span: int=8,
        seg_emb: bool=False,
        text_masking: bool=False,
        pad_value: int=0,
        not_sequence: Collection[str]=(),
) -> Tuple[List[str], Dict[str, paddle.Tensor]]:
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(not k.endswith("_lens")
               for k in data[0]), f"*_lens is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [paddle.to_tensor(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = paddle.to_tensor(
                [d[key].shape[0] for d in data], dtype=paddle.int64)
            output[key + "_lens"] = lens

    feats = feats_extract.get_log_mel_fbank(np.array(output["speech"][0]))
    feats = paddle.to_tensor(feats)
    print("feats.shape:", feats.shape)
    feats_lens = paddle.shape(feats)[0]
    feats = paddle.unsqueeze(feats, 0)

    text = output["text"]
    text_lens = output["text_lens"]
    align_start = output["align_start"]
    align_start_lens = output["align_start_lens"]
    align_end = output["align_end"]

    max_tlen = max(text_lens)
    max_slen = max(feats_lens)

    speech_pad = feats[:, :max_slen]

    text_pad = text
    text_mask = make_non_pad_mask(
        text_lens, text_pad, length_dim=1).unsqueeze(-2)
    speech_mask = make_non_pad_mask(
        feats_lens, speech_pad[:, :, 0], length_dim=1).unsqueeze(-2)

    span_bdy = None
    if 'span_bdy' in output.keys():
        span_bdy = output['span_bdy']

    # dual_mask 的是混合中英时候同时 mask 语音和文本 
    # ernie sat 在实现跨语言的时候都 mask 了
    if text_masking:
        masked_pos, text_masked_pos = phones_text_masking(
            xs_pad=speech_pad,
            src_mask=speech_mask,
            text_pad=text_pad,
            text_mask=text_mask,
            align_start=align_start,
            align_end=align_end,
            align_start_lens=align_start_lens,
            mlm_prob=mlm_prob,
            mean_phn_span=mean_phn_span,
            span_bdy=span_bdy)
    # 训练纯中文和纯英文的 -> a3t 没有对 phoneme 做 mask, 只对语音 mask 了
    # a3t 和 ernie sat 的区别主要在于做 mask 的时候
    else:
        masked_pos = phones_masking(
            xs_pad=speech_pad,
            src_mask=speech_mask,
            align_start=align_start,
            align_end=align_end,
            align_start_lens=align_start_lens,
            mlm_prob=mlm_prob,
            mean_phn_span=mean_phn_span,
            span_bdy=span_bdy)
        text_masked_pos = paddle.zeros(paddle.shape(text_pad))

    output_dict = {}

    speech_seg_pos, text_seg_pos = get_seg_pos(
        speech_pad=speech_pad,
        text_pad=text_pad,
        align_start=align_start,
        align_end=align_end,
        align_start_lens=align_start_lens,
        seg_emb=seg_emb)
    output_dict['speech'] = speech_pad
    output_dict['text'] = text_pad
    output_dict['masked_pos'] = masked_pos
    output_dict['text_masked_pos'] = text_masked_pos
    output_dict['speech_mask'] = speech_mask
    output_dict['text_mask'] = text_mask
    output_dict['speech_seg_pos'] = speech_seg_pos
    output_dict['text_seg_pos'] = text_seg_pos
    output = (uttids, output_dict)
    return output


def build_mlm_collate_fn(
        sr: int=24000,
        n_fft: int=2048,
        hop_length: int=300,
        win_length: int=None,
        n_mels: int=80,
        fmin: int=80,
        fmax: int=7600,
        mlm_prob: float=0.8,
        mean_phn_span: int=8,
        seg_emb: bool=False,
        epoch: int=-1, ):
    feats_extract_class = LogMelFBank

    feats_extract = feats_extract_class(
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax)

    if epoch == -1:
        mlm_prob_factor = 1
    else:
        mlm_prob_factor = 0.8

    return MLMCollateFn(
        feats_extract=feats_extract,
        mlm_prob=mlm_prob * mlm_prob_factor,
        mean_phn_span=mean_phn_span,
        seg_emb=seg_emb)
