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
import numpy as np
import paddle

from paddlespeech.t2s.data.batch import batch_sequences


def speedyspeech_batch_fn(examples):
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
