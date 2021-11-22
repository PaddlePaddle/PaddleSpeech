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
import argparse
import os
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
from matplotlib import pyplot as plt

from paddlespeech.t2s.exps.voice_cloning.tacotron2_ge2e.aishell3 import voc_phones
from paddlespeech.t2s.exps.voice_cloning.tacotron2_ge2e.aishell3 import voc_tones
from paddlespeech.t2s.exps.voice_cloning.tacotron2_ge2e.chinese_g2p import convert_sentence
from paddlespeech.t2s.models.tacotron2 import Tacotron2
from paddlespeech.t2s.models.waveflow import ConditionalWaveFlow
from paddlespeech.t2s.utils import display
from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor
from paddlespeech.vector.models.lstm_speaker_encoder import LSTMSpeakerEncoder


def voice_cloning(args):
    # speaker encoder
    p = SpeakerVerificationPreprocessor(
        sampling_rate=16000,
        audio_norm_target_dBFS=-30,
        vad_window_length=30,
        vad_moving_average_width=8,
        vad_max_silence_length=6,
        mel_window_length=25,
        mel_window_step=10,
        n_mels=40,
        partial_n_frames=160,
        min_pad_coverage=0.75,
        partial_overlap_ratio=0.5)
    print("Audio Processor Done!")

    speaker_encoder = LSTMSpeakerEncoder(
        n_mels=40, num_layers=3, hidden_size=256, output_size=256)
    speaker_encoder.set_state_dict(paddle.load(args.ge2e_params_path))
    speaker_encoder.eval()
    print("GE2E Done!")

    synthesizer = Tacotron2(
        vocab_size=68,
        n_tones=10,
        d_mels=80,
        d_encoder=512,
        encoder_conv_layers=3,
        encoder_kernel_size=5,
        d_prenet=256,
        d_attention_rnn=1024,
        d_decoder_rnn=1024,
        attention_filters=32,
        attention_kernel_size=31,
        d_attention=128,
        d_postnet=512,
        postnet_kernel_size=5,
        postnet_conv_layers=5,
        reduction_factor=1,
        p_encoder_dropout=0.5,
        p_prenet_dropout=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        p_postnet_dropout=0.5,
        d_global_condition=256,
        use_stop_token=False, )
    synthesizer.set_state_dict(paddle.load(args.tacotron2_params_path))
    synthesizer.eval()
    print("Tacotron2 Done!")

    # vocoder
    vocoder = ConditionalWaveFlow(
        upsample_factors=[16, 16],
        n_flows=8,
        n_layers=8,
        n_group=16,
        channels=128,
        n_mels=80,
        kernel_size=[3, 3])
    vocoder.set_state_dict(paddle.load(args.waveflow_params_path))
    vocoder.eval()
    print("WaveFlow Done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)

    # 因为 AISHELL-3 数据集中使用 % 和 $ 表示韵律词和韵律短语的边界，它们大约对应着较短和较长的停顿，在文本中可以使用 % 和 $ 来调节韵律。
    # 值得的注意的是，句子的有效字符集仅包含汉字和 %, $, 因此输入的句子只能包含这些字符。
    sentence = "每当你觉得%想要批评什么人的时候$你切要记着%这个世界上的人%并非都具备你禀有的条件$"
    phones, tones = convert_sentence(sentence)
    phones = np.array(
        [voc_phones.lookup(item) for item in phones], dtype=np.int64)
    tones = np.array([voc_tones.lookup(item) for item in tones], dtype=np.int64)
    phones = paddle.to_tensor(phones).unsqueeze(0)
    tones = paddle.to_tensor(tones).unsqueeze(0)

    for name in os.listdir(input_dir):
        utt_id = name.split(".")[0]
        ref_audio_path = input_dir / name
        mel_sequences = p.extract_mel_partials(p.preprocess_wav(ref_audio_path))
        print("mel_sequences: ", mel_sequences.shape)
        with paddle.no_grad():
            embed = speaker_encoder.embed_utterance(
                paddle.to_tensor(mel_sequences))
        print("embed shape: ", embed.shape)
        utterance_embeds = paddle.unsqueeze(embed, 0)
        outputs = synthesizer.infer(
            phones, tones=tones, global_condition=utterance_embeds)
        mel_input = paddle.transpose(outputs["mel_outputs_postnet"], [0, 2, 1])
        alignment = outputs["alignments"][0].numpy().T
        display.plot_alignment(alignment)
        plt.savefig(str(output_dir / (utt_id + ".png")))

        with paddle.no_grad():
            wav = vocoder.infer(mel_input)
        wav = wav.numpy()[0]
        sf.write(str(output_dir / (utt_id + ".wav")), wav, samplerate=22050)


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--ge2e_params_path", type=str, help="ge2e params path.")
    parser.add_argument(
        "--tacotron2_params_path", type=str, help="tacotron2 params path.")
    parser.add_argument(
        "--waveflow_params_path", type=str, help="waveflow params path.")

    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu=0, use cpu.")

    parser.add_argument(
        "--input-dir",
        type=str,
        help="input dir of *.wav, the sample rate will be resample to 16k.")
    parser.add_argument("--output-dir", type=str, help="output dir.")

    args = parser.parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    voice_cloning(args)


if __name__ == "__main__":
    main()
