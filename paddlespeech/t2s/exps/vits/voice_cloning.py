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

import librosa
import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.get_feats import LinearSpectrogram
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.models.vits import VITS
from paddlespeech.t2s.utils import str2bool
from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor
from paddlespeech.vector.models.lstm_speaker_encoder import LSTMSpeakerEncoder


def voice_cloning(args):

    # Init body.
    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)

    # speaker encoder
    spec_extractor = LinearSpectrogram(n_fft=config.n_fft,
                                       hop_length=config.n_shift,
                                       win_length=config.win_length,
                                       window=config.window)
    p = SpeakerVerificationPreprocessor(sampling_rate=16000,
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

    speaker_encoder = LSTMSpeakerEncoder(n_mels=40,
                                         num_layers=3,
                                         hidden_size=256,
                                         output_size=256)
    speaker_encoder.set_state_dict(paddle.load(args.ge2e_params_path))
    speaker_encoder.eval()
    print("GE2E Done!")

    frontend = get_frontend(lang=args.lang, phones_dict=args.phones_dict)
    print("frontend done!")

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    odim = config.n_fft // 2 + 1

    vits = VITS(idim=vocab_size, odim=odim, **config["model"])
    vits.set_state_dict(paddle.load(args.ckpt)["main_params"])
    vits.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)

    if args.audio_path == "":
        args.audio_path = None
    if args.audio_path is None:
        sentence = args.text
        merge_sentences = True
        add_blank = args.add_blank

        if args.lang == 'zh':
            input_ids = frontend.get_input_ids(sentence,
                                               merge_sentences=merge_sentences,
                                               add_blank=add_blank)
        elif args.lang == 'en':
            input_ids = frontend.get_input_ids(sentence,
                                               merge_sentences=merge_sentences)
        phone_ids = input_ids["phone_ids"][0]
    else:
        wav, _ = librosa.load(str(args.audio_path), sr=config.fs)
        feats = paddle.to_tensor(spec_extractor.get_linear_spectrogram(wav))

        mel_sequences = p.extract_mel_partials(p.preprocess_wav(
            args.audio_path))
        with paddle.no_grad():
            spk_emb_src = speaker_encoder.embed_utterance(
                paddle.to_tensor(mel_sequences))

    for name in os.listdir(input_dir):
        utt_id = name.split(".")[0]
        ref_audio_path = input_dir / name
        mel_sequences = p.extract_mel_partials(p.preprocess_wav(ref_audio_path))
        # print("mel_sequences: ", mel_sequences.shape)
        with paddle.no_grad():
            spk_emb = speaker_encoder.embed_utterance(
                paddle.to_tensor(mel_sequences))
        # print("spk_emb shape: ", spk_emb.shape)

        with paddle.no_grad():
            if args.audio_path is None:
                out = vits.inference(text=phone_ids, spembs=spk_emb)
            else:
                out = vits.voice_conversion(feats=feats,
                                            spembs_src=spk_emb_src,
                                            spembs_tgt=spk_emb)
            wav = out["wav"]

        sf.write(str(output_dir / (utt_id + ".wav")),
                 wav.numpy(),
                 samplerate=config.fs)
        print(f"{utt_id} done!")
    # Randomly generate numbers of 0 ~ 0.2, 256 is the dim of spk_emb
    random_spk_emb = np.random.rand(256) * 0.2
    random_spk_emb = paddle.to_tensor(random_spk_emb, dtype='float32')
    utt_id = "random_spk_emb"
    with paddle.no_grad():
        if args.audio_path is None:
            out = vits.inference(text=phone_ids, spembs=random_spk_emb)
        else:
            out = vits.voice_conversion(feats=feats,
                                        spembs_src=spk_emb_src,
                                        spembs_tgt=random_spk_emb)
        wav = out["wav"]
    sf.write(str(output_dir / (utt_id + ".wav")),
             wav.numpy(),
             samplerate=config.fs)
    print(f"{utt_id} done!")


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config',
                        type=str,
                        default=None,
                        help='Config of VITS.')
    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        help='Checkpoint file of VITS.')
    parser.add_argument("--phones_dict",
                        type=str,
                        default=None,
                        help="phone vocabulary file.")
    parser.add_argument("--text",
                        type=str,
                        default="每当你觉得，想要批评什么人的时候，你切要记着，这个世界上的人，并非都具备你禀有的条件。",
                        help="text to synthesize, a line")
    parser.add_argument('--lang',
                        type=str,
                        default='zh',
                        help='Choose model language. zh or en')
    parser.add_argument("--audio-path",
                        type=str,
                        default=None,
                        help="audio as content to synthesize")

    parser.add_argument("--ge2e_params_path",
                        type=str,
                        help="ge2e params path.")

    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="if ngpu=0, use cpu.")

    parser.add_argument(
        "--input-dir",
        type=str,
        help="input dir of *.wav, the sample rate will be resample to 16k.")
    parser.add_argument("--output-dir", type=str, help="output dir.")

    parser.add_argument("--add-blank",
                        type=str2bool,
                        default=True,
                        help="whether to add blank between phones")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    voice_cloning(args)


if __name__ == "__main__":
    main()
