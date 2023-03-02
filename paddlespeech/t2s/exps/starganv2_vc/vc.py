# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import time
from pathlib import Path

import librosa
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.cli.utils import download_and_decompress
from paddlespeech.resource.pretrained_models import StarGANv2VC_source
from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator
from paddlespeech.t2s.models.starganv2_vc import Generator
from paddlespeech.t2s.models.starganv2_vc import JDCNet
from paddlespeech.t2s.models.starganv2_vc import MappingNetwork
from paddlespeech.t2s.models.starganv2_vc import StyleEncoder
from paddlespeech.utils.env import MODEL_HOME


def get_mel_extractor():
    sr = 16000
    n_fft = 2048
    win_length = 1200
    hop_length = 300
    n_mels = 80
    fmin = 0
    fmax = sr // 2

    mel_extractor = LogMelFBank(
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        norm=None,
        htk=True,
        power=2.0)

    return mel_extractor


def preprocess(wave, mel_extractor):
    logmel = mel_extractor.get_log_mel_fbank(wave, base='e')
    # [1, 80, 1011]
    mean, std = -4, 4
    mel_tensor = (paddle.to_tensor(logmel.T).unsqueeze(0) - mean) / std
    return mel_tensor


def compute_style(speaker_dicts, mel_extractor, style_encoder, mapping_network):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == '':
            label = paddle.to_tensor([speaker], dtype=paddle.int64)
            latent_dim = mapping_network.shared[0].weight.shape[0]
            ref = mapping_network(paddle.randn([1, latent_dim]), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave, mel_extractor)

            with paddle.no_grad():
                label = paddle.to_tensor([speaker], dtype=paddle.int64)
                ref = style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)

    return reference_embeddings


def get_models(args, uncompress_path):
    model_dict = {}
    jdc_model_dir = os.path.join(uncompress_path, 'jdcnet.pdz')
    voc_model_dir = os.path.join(uncompress_path, 'Vocoder/')
    starganv2vc_model_dir = os.path.join(uncompress_path, 'starganv2vc.pdz')

    F0_model = JDCNet(num_class=1, seq_len=192)
    F0_model.set_state_dict(paddle.load(jdc_model_dir)['main_params'])
    F0_model.eval()

    voc_config_path = os.path.join(voc_model_dir, 'config.yml')
    with open(voc_config_path) as f:
        voc_config = CfgNode(yaml.safe_load(f))
    voc_config["generator_params"].pop("upsample_net")
    voc_config["generator_params"]["upsample_scales"] = voc_config[
        "generator_params"].pop("upsample_params")["upsample_scales"]
    vocoder = PWGGenerator(**voc_config["generator_params"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    voc_model_path = os.path.join(voc_model_dir, 'checkpoint-400000steps.pd')
    vocoder.set_state_dict(paddle.load(voc_model_path))

    with open(args.config_path) as f:
        config = CfgNode(yaml.safe_load(f))

    generator = Generator(**config['generator_params'])
    mapping_network = MappingNetwork(**config['mapping_network_params'])
    style_encoder = StyleEncoder(**config['style_encoder_params'])

    starganv2vc_model_param = paddle.load(starganv2vc_model_dir)

    generator.set_state_dict(starganv2vc_model_param['generator_params'])
    mapping_network.set_state_dict(
        starganv2vc_model_param['mapping_network_params'])
    style_encoder.set_state_dict(
        starganv2vc_model_param['style_encoder_params'])

    generator.eval()
    mapping_network.eval()
    style_encoder.eval()

    model_dict['F0_model'] = F0_model
    model_dict['vocoder'] = vocoder
    model_dict['generator'] = generator
    model_dict['mapping_network'] = mapping_network
    model_dict['style_encoder'] = style_encoder
    return model_dict


def voice_conversion(args, uncompress_path):
    speakers = [
        225, 228, 229, 230, 231, 233, 236, 239, 240, 244, 226, 227, 232, 243,
        254, 256, 258, 259, 270, 273
    ]
    demo_dir = os.path.join(uncompress_path, 'Demo/VCTK-corpus/')
    model_dict = get_models(args, uncompress_path=uncompress_path)
    style_encoder = model_dict['style_encoder']
    mapping_network = model_dict['mapping_network']
    generator = model_dict['generator']
    vocoder = model_dict['vocoder']
    F0_model = model_dict['F0_model']

    # 计算 Demo 文件夹下的说话人的风格
    speaker_dicts = {}
    selected_speakers = [273, 259, 258, 243, 254, 244, 236, 233, 230, 228]
    for s in selected_speakers:
        k = s
        speaker_dicts['p' + str(s)] = (
            demo_dir + 'p' + str(k) + '/p' + str(k) + '_023.wav',
            speakers.index(s))
    mel_extractor = get_mel_extractor()
    reference_embeddings = compute_style(
        speaker_dicts=speaker_dicts,
        mel_extractor=mel_extractor,
        style_encoder=style_encoder,
        mapping_network=mapping_network)

    wave, sr = librosa.load(args.source_path, sr=24000)
    source = preprocess(wave, mel_extractor)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_wav_name = str(output_dir / 'orig_voc.wav')
    print('原始语音 (使用声码器解码): %s' % orig_wav_name)
    c = source.transpose([0, 2, 1]).squeeze()
    with paddle.no_grad():
        recon = vocoder.inference(c)
        recon = recon.reshape([-1]).numpy()
    sf.write(orig_wav_name, recon, samplerate=24000)

    keys = []
    converted_samples = {}
    reconstructed_samples = {}
    converted_mels = {}
    start = time.time()

    for key, (ref, _) in reference_embeddings.items():
        with paddle.no_grad():
            # F0_model 输入的特征是否可以不带 norm，或者 norm 是否一定要和 stargan 原作保持一致？
            # !! 需要，ASR 和 F0_model 用的是一样的数据预处理方式
            # 如果不想要重新训练 ASR 和 F0_model, 则我们的数据预处理需要和 stargan 原作保持一致
            # 但是 vocoder 就无法复用
            # 是否因为 asr 的输入是 16k 的，所以 torchaudio 的参数也是 16k 的？
            f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
            # 输出是带 norm 的 mel, 所以可以直接用 vocoder.inference
            out = generator(source.unsqueeze(1), ref, F0=f0_feat)
            c = out.transpose([0, 1, 3, 2]).squeeze()
            y_out = vocoder.inference(c)
            y_out = y_out.reshape([-1])
            if key not in speaker_dicts or speaker_dicts[key][0] == "":
                recon = None
            else:
                wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
                mel = preprocess(wave, mel_extractor)
                c = mel.transpose([0, 2, 1]).squeeze()
                recon = vocoder.inference(c)
                recon = recon.reshape([-1]).numpy()

        converted_samples[key] = y_out.numpy()
        reconstructed_samples[key] = recon
        converted_mels[key] = out
        keys.append(key)
    end = time.time()
    print('总共花费时间: %.3f sec' % (end - start))
    for key, wave in converted_samples.items():
        wav_name = str(output_dir / ('vc_result_' + key + '.wav'))
        print('语音转换结果: %s' % wav_name)
        sf.write(wav_name, wave, samplerate=24000)
        ref_wav_name = str(output_dir / ('ref_voc_' + key + '.wav'))
        print('参考的说话人 (使用声码器解码): %s' % ref_wav_name)
        if reconstructed_samples[key] is not None:
            sf.write(ref_wav_name, reconstructed_samples[key], samplerate=24000)


def parse_args():
    # parse args and config  
    parser = argparse.ArgumentParser(
        description="StarGANv2-VC Voice Conversion.")
    parser.add_argument("--source_path", type=str, help="source audio's path.")
    parser.add_argument("--output_dir", type=str, help="output dir.")
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help='Config of StarGANv2-VC model.')
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
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
    model_version = '1.0'
    uncompress_path = download_and_decompress(StarGANv2VC_source[model_version],
                                              MODEL_HOME)
    voice_conversion(args, uncompress_path=uncompress_path)


if __name__ == "__main__":
    main()
