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

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.modules.normalizer import ZScore
from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor
from paddlespeech.vector.models.lstm_speaker_encoder import LSTMSpeakerEncoder

model_alias = {
    # acoustic model
    "fastspeech2":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2",
    "fastspeech2_inference":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2Inference",
    "tacotron2":
    "paddlespeech.t2s.models.tacotron2:Tacotron2",
    "tacotron2_inference":
    "paddlespeech.t2s.models.tacotron2:Tacotron2Inference",
    # voc
    "pwgan":
    "paddlespeech.t2s.models.parallel_wavegan:PWGGenerator",
    "pwgan_inference":
    "paddlespeech.t2s.models.parallel_wavegan:PWGInference",
}


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

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    # acoustic model
    odim = am_config.n_mels
    # model: {model_name}_{dataset}
    am_name = args.am[:args.am.rindex('_')]
    am_dataset = args.am[args.am.rindex('_') + 1:]

    am_class = dynamic_import(am_name, model_alias)
    am_inference_class = dynamic_import(am_name + '_inference', model_alias)

    if am_name == 'fastspeech2':
        am = am_class(
            idim=vocab_size, odim=odim, spk_num=None, **am_config["model"])
    elif am_name == 'tacotron2':
        am = am_class(idim=vocab_size, odim=odim, **am_config["model"])

    am.set_state_dict(paddle.load(args.am_ckpt)["main_params"])
    am.eval()
    am_mu, am_std = np.load(args.am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)
    am_normalizer = ZScore(am_mu, am_std)
    am_inference = am_inference_class(am_normalizer, am)
    am_inference.eval()
    print("acoustic model done!")

    # vocoder
    # model: {model_name}_{dataset}
    voc_name = args.voc[:args.voc.rindex('_')]
    voc_class = dynamic_import(voc_name, model_alias)
    voc_inference_class = dynamic_import(voc_name + '_inference', model_alias)
    voc = voc_class(**voc_config["generator_params"])
    voc.set_state_dict(paddle.load(args.voc_ckpt)["generator_params"])
    voc.remove_weight_norm()
    voc.eval()
    voc_mu, voc_std = np.load(args.voc_stat)
    voc_mu = paddle.to_tensor(voc_mu)
    voc_std = paddle.to_tensor(voc_std)
    voc_normalizer = ZScore(voc_mu, voc_std)
    voc_inference = voc_inference_class(voc_normalizer, voc)
    voc_inference.eval()
    print("voc done!")

    frontend = Frontend(phone_vocab_path=args.phones_dict)
    print("frontend done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)

    sentence = args.text

    input_ids = frontend.get_input_ids(sentence, merge_sentences=True)
    phone_ids = input_ids["phone_ids"][0]

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
            wav = voc_inference(am_inference(phone_ids, spk_emb=spk_emb))

        sf.write(
            str(output_dir / (utt_id + ".wav")),
            wav.numpy(),
            samplerate=am_config.fs)
        print(f"{utt_id} done!")
    # Randomly generate numbers of 0 ~ 0.2, 256 is the dim of spk_emb
    random_spk_emb = np.random.rand(256) * 0.2
    random_spk_emb = paddle.to_tensor(random_spk_emb)
    utt_id = "random_spk_emb"
    with paddle.no_grad():
        wav = voc_inference(am_inference(phone_ids, spk_emb=spk_emb))
    sf.write(
        str(output_dir / (utt_id + ".wav")),
        wav.numpy(),
        samplerate=am_config.fs)
    print(f"{utt_id} done!")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=['fastspeech2_aishell3', 'tacotron2_aishell3'],
        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        '--am_config',
        type=str,
        default=None,
        help='Config of acoustic model. Use deault config when it is None.')
    parser.add_argument(
        '--am_ckpt',
        type=str,
        default=None,
        help='Checkpoint file of acoustic model.')
    parser.add_argument(
        "--am_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training acoustic model."
    )
    parser.add_argument(
        "--phones-dict",
        type=str,
        default="phone_id_map.txt",
        help="phone vocabulary file.")
    # vocoder
    parser.add_argument(
        '--voc',
        type=str,
        default='pwgan_csmsc',
        choices=['pwgan_aishell3'],
        help='Choose vocoder type of tts task.')

    parser.add_argument(
        '--voc_config',
        type=str,
        default=None,
        help='Config of voc. Use deault config when it is None.')
    parser.add_argument(
        '--voc_ckpt', type=str, default=None, help='Checkpoint file of voc.')
    parser.add_argument(
        "--voc_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training voc."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="每当你觉得，想要批评什么人的时候，你切要记着，这个世界上的人，并非都具备你禀有的条件。",
        help="text to synthesize, a line")

    parser.add_argument(
        "--ge2e_params_path", type=str, help="ge2e params path.")

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
