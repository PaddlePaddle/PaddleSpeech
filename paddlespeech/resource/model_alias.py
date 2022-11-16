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

__all__ = [
    'model_alias',
]

# Records of model name to import class
model_alias = {
    # ---------------------------------
    # -------------- SSL --------------
    # ---------------------------------
    "wav2vec2ASR": ["paddlespeech.s2t.models.wav2vec2:Wav2vec2ASR"],
    "wav2vec2": ["paddlespeech.s2t.models.wav2vec2:Wav2vec2Base"],

    # ---------------------------------
    # -------------- ASR --------------
    # ---------------------------------
    "deepspeech2offline": ["paddlespeech.s2t.models.ds2:DeepSpeech2Model"],
    "deepspeech2online": ["paddlespeech.s2t.models.ds2:DeepSpeech2Model"],
    "conformer": ["paddlespeech.s2t.models.u2:U2Model"],
    "conformer_online": ["paddlespeech.s2t.models.u2:U2Model"],
    "conformer_u2pp_online": ["paddlespeech.s2t.models.u2:U2Model"],
    "transformer": ["paddlespeech.s2t.models.u2:U2Model"],
    "wenetspeech": ["paddlespeech.s2t.models.u2:U2Model"],

    # ---------------------------------
    # -------------- CLS --------------
    # ---------------------------------
    "panns_cnn6": ["paddlespeech.cls.models.panns:CNN6"],
    "panns_cnn10": ["paddlespeech.cls.models.panns:CNN10"],
    "panns_cnn14": ["paddlespeech.cls.models.panns:CNN14"],

    # ---------------------------------
    # -------------- ST ---------------
    # ---------------------------------
    "fat_st": ["paddlespeech.s2t.models.u2_st:U2STModel"],

    # ---------------------------------
    # -------------- TEXT -------------
    # ---------------------------------
    "ernie_linear_p7": [
        "paddlespeech.text.models:ErnieLinear",
        "paddlenlp.transformers:ErnieTokenizer"
    ],
    "ernie_linear_p3": [
        "paddlespeech.text.models:ErnieLinear",
        "paddlenlp.transformers:ErnieTokenizer"
    ],
    "ernie_linear_p3_wudao": [
        "paddlespeech.text.models:ErnieLinear",
        "paddlenlp.transformers:ErnieTokenizer"
    ],

    # ---------------------------------
    # -------------- TTS --------------
    # ---------------------------------
    # acoustic model
    "speedyspeech": ["paddlespeech.t2s.models.speedyspeech:SpeedySpeech"],
    "speedyspeech_inference":
    ["paddlespeech.t2s.models.speedyspeech:SpeedySpeechInference"],
    "fastspeech2": ["paddlespeech.t2s.models.fastspeech2:FastSpeech2"],
    "fastspeech2_inference":
    ["paddlespeech.t2s.models.fastspeech2:FastSpeech2Inference"],
    "tacotron2": ["paddlespeech.t2s.models.tacotron2:Tacotron2"],
    "tacotron2_inference":
    ["paddlespeech.t2s.models.tacotron2:Tacotron2Inference"],
    # voc
    "pwgan": ["paddlespeech.t2s.models.parallel_wavegan:PWGGenerator"],
    "pwgan_inference":
    ["paddlespeech.t2s.models.parallel_wavegan:PWGInference"],
    "mb_melgan": ["paddlespeech.t2s.models.melgan:MelGANGenerator"],
    "mb_melgan_inference": ["paddlespeech.t2s.models.melgan:MelGANInference"],
    "style_melgan": ["paddlespeech.t2s.models.melgan:StyleMelGANGenerator"],
    "style_melgan_inference":
    ["paddlespeech.t2s.models.melgan:StyleMelGANInference"],
    "hifigan": ["paddlespeech.t2s.models.hifigan:HiFiGANGenerator"],
    "hifigan_inference": ["paddlespeech.t2s.models.hifigan:HiFiGANInference"],
    "wavernn": ["paddlespeech.t2s.models.wavernn:WaveRNN"],
    "wavernn_inference": ["paddlespeech.t2s.models.wavernn:WaveRNNInference"],

    # ---------------------------------
    # ------------ Vector -------------
    # ---------------------------------
    "ecapatdnn": ["paddlespeech.vector.models.ecapa_tdnn:EcapaTdnn"],

    # ---------------------------------
    # -------------- kws --------------
    # ---------------------------------
    "mdtc": ["paddlespeech.kws.models.mdtc:MDTC"],
    "mdtc_for_kws": ["paddlespeech.kws.models.mdtc:KWSModel"],
}
