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
import yaml
from yacs.config import CfgNode

from paddlespeech.cli.vector import VectorExecutor
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.utils import str2bool
from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor
from paddlespeech.vector.models.lstm_speaker_encoder import LSTMSpeakerEncoder


def gen_random_embed(use_ecapa: bool = False):
    if use_ecapa:
        # Randomly generate numbers of -25 ~ 25, 192 is the dim of spk_emb
        random_spk_emb = (-1 + 2 * np.random.rand(192)) * 25

    # GE2E
    else:
        # Randomly generate numbers of 0 ~ 0.2, 256 is the dim of spk_emb
        random_spk_emb = np.random.rand(256) * 0.2
    random_spk_emb = paddle.to_tensor(random_spk_emb, dtype='float32')
    return random_spk_emb


def voice_cloning(args):
    # Init body.
    with open(args.am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(am_config)
    print(voc_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)

    # speaker encoder
    if args.use_ecapa:
        vec_executor = VectorExecutor()
        # warm up
        vec_executor(audio_file=input_dir / os.listdir(input_dir)[0],
                     force_yes=True)
        print("ECAPA-TDNN Done!")
    # use GE2E
    else:
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

    frontend = Frontend(phone_vocab_path=args.phones_dict)
    print("frontend done!")

    sentence = args.text
    input_ids = frontend.get_input_ids(sentence, merge_sentences=True)
    phone_ids = input_ids["phone_ids"][0]

    # acoustic model
    am_inference = get_am_inference(am=args.am,
                                    am_config=am_config,
                                    am_ckpt=args.am_ckpt,
                                    am_stat=args.am_stat,
                                    phones_dict=args.phones_dict)

    # vocoder
    voc_inference = get_voc_inference(voc=args.voc,
                                      voc_config=voc_config,
                                      voc_ckpt=args.voc_ckpt,
                                      voc_stat=args.voc_stat)

    for name in os.listdir(input_dir):
        utt_id = name.split(".")[0]
        ref_audio_path = input_dir / name
        if args.use_ecapa:
            spk_emb = vec_executor(audio_file=ref_audio_path, force_yes=True)
            spk_emb = paddle.to_tensor(spk_emb)
        # GE2E
        else:
            mel_sequences = p.extract_mel_partials(
                p.preprocess_wav(ref_audio_path))
            with paddle.no_grad():
                spk_emb = speaker_encoder.embed_utterance(
                    paddle.to_tensor(mel_sequences))
        with paddle.no_grad():
            wav = voc_inference(am_inference(phone_ids, spk_emb=spk_emb))

        sf.write(str(output_dir / (utt_id + ".wav")),
                 wav.numpy(),
                 samplerate=am_config.fs)
        print(f"{utt_id} done!")

    # generate 5 random_spk_emb
    # for i in range(5):
    #     random_spk_emb = gen_random_embed(args.use_ecapa)
    #     utt_id = "random_spk_emb"
    #     with paddle.no_grad():
    #         wav = voc_inference(am_inference(phone_ids, spk_emb=random_spk_emb))
    #     sf.write(
    #         str(output_dir / (utt_id + "_" + str(i) + ".wav")),
    #         wav.numpy(),
    #         samplerate=am_config.fs)
    # print(f"{utt_id} done!")


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--am',
                        type=str,
                        default='fastspeech2_csmsc',
                        choices=['fastspeech2_aishell3', 'tacotron2_aishell3'],
                        help='Choose acoustic model type of tts task.')
    parser.add_argument('--am_config',
                        type=str,
                        default=None,
                        help='Config of acoustic model.')
    parser.add_argument('--am_ckpt',
                        type=str,
                        default=None,
                        help='Checkpoint file of acoustic model.')
    parser.add_argument(
        "--am_stat",
        type=str,
        default=None,
        help=
        "mean and standard deviation used to normalize spectrogram when training acoustic model."
    )
    parser.add_argument("--phones-dict",
                        type=str,
                        default="phone_id_map.txt",
                        help="phone vocabulary file.")
    # vocoder
    parser.add_argument('--voc',
                        type=str,
                        default='pwgan_csmsc',
                        choices=['pwgan_aishell3'],
                        help='Choose vocoder type of tts task.')

    parser.add_argument('--voc_config',
                        type=str,
                        default=None,
                        help='Config of voc.')
    parser.add_argument('--voc_ckpt',
                        type=str,
                        default=None,
                        help='Checkpoint file of voc.')
    parser.add_argument(
        "--voc_stat",
        type=str,
        default=None,
        help=
        "mean and standard deviation used to normalize spectrogram when training voc."
    )
    parser.add_argument("--text",
                        type=str,
                        default="每当你觉得，想要批评什么人的时候，你切要记着，这个世界上的人，并非都具备你禀有的条件。",
                        help="text to synthesize, a line")
    parser.add_argument("--ge2e_params_path",
                        type=str,
                        help="ge2e params path.")
    parser.add_argument("--use_ecapa",
                        type=str2bool,
                        default=False,
                        help="whether to use ECAPA-TDNN as speaker encoder.")
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="if ngpu=0, use cpu.")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="input dir of *.wav, the sample rate will be resample to 16k.")
    parser.add_argument("--output-dir", type=str, help="output dir.")

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
