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
import time

import librosa
import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator
from paddlespeech.t2s.models.starganv2_vc import Generator
from paddlespeech.t2s.models.starganv2_vc import JDCNet
from paddlespeech.t2s.models.starganv2_vc import MappingNetwork
from paddlespeech.t2s.models.starganv2_vc import StyleEncoder

jdc_modeldir = '/home/yuantian01/PaddleSpeech_stargan/PaddleSpeech/stargan_models/jdcnet.pdz'
# 是 stargan 重新训练的
voc_modeldir = '/home/yuantian01/PaddleSpeech_stargan/PaddleSpeech/stargan_models/Vocoder/'
starganv2vc_modeldir = '/home/yuantian01/PaddleSpeech_stargan/PaddleSpeech/stargan_models/starganv2vc.pdz'

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

speakers = [
    225, 228, 229, 230, 231, 233, 236, 239, 240, 244, 226, 227, 232, 243, 254,
    256, 258, 259, 270, 273
]

mean, std = -4, 4


def preprocess(wave):
    logmel = mel_extractor.get_log_mel_fbank(wave, base='e')
    # [1, 80, 1011]
    mel_tensor = (paddle.to_tensor(logmel.T).unsqueeze(0) - mean) / std
    return mel_tensor


def compute_style(speaker_dicts):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == "":
            label = paddle.to_tensor([speaker], dtype=paddle.int64)
            latent_dim = mapping_network.shared[0].weight.shape[0]
            ref = mapping_network(paddle.randn([1, latent_dim]), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave)

            with paddle.no_grad():
                label = paddle.to_tensor([speaker], dtype=paddle.int64)
                ref = style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)

    return reference_embeddings


F0_model = JDCNet(num_class=1, seq_len=192)
i = 0

F0_model.set_state_dict(paddle.load(jdc_modeldir)['main_params'])
F0_model.eval()

with open(voc_modeldir + 'config.yml') as f:
    voc_config = CfgNode(yaml.safe_load(f))
voc_config["generator_params"].pop("upsample_net")
voc_config["generator_params"]["upsample_scales"] = voc_config[
    "generator_params"].pop("upsample_params")["upsample_scales"]
vocoder = PWGGenerator(**voc_config["generator_params"])
vocoder.remove_weight_norm()
vocoder.eval()
vocoder.set_state_dict(paddle.load(voc_modeldir + 'checkpoint-400000steps.pd'))

dim_in = 64
style_dim = 64
latent_dim = 16
num_domains = 20
max_conv_dim = 512
n_repeat = 4
w_hpf = 0
F0_channel = 256

generator = Generator(
    dim_in=dim_in,
    style_dim=style_dim,
    max_conv_dim=max_conv_dim,
    w_hpf=w_hpf,
    F0_channel=F0_channel)
mapping_network = MappingNetwork(
    latent_dim=latent_dim,
    style_dim=style_dim,
    num_domains=num_domains,
    hidden_dim=max_conv_dim)
style_encoder = StyleEncoder(
    dim_in=dim_in,
    style_dim=style_dim,
    num_domains=num_domains,
    max_conv_dim=max_conv_dim)

starganv2vc_model_param = paddle.load(starganv2vc_modeldir)
generator.set_state_dict(starganv2vc_model_param['generator_params'])
mapping_network.set_state_dict(
    starganv2vc_model_param['mapping_network_params'])
style_encoder.set_state_dict(starganv2vc_model_param['style_encoder_params'])
generator.eval()
mapping_network.eval()
style_encoder.eval()

# 计算Demo文件夹下的说话人的风格
speaker_dicts = {}
selected_speakers = [273, 259, 258, 243, 254, 244, 236, 233, 230, 228]
for s in selected_speakers:
    k = s
    speaker_dicts['p' + str(s)] = (
        'Demo/VCTK-corpus/p' + str(k) + '/p' + str(k) + '_023.wav',
        speakers.index(s))
print("speaker_dicts:", speaker_dicts)
reference_embeddings = compute_style(speaker_dicts)
# print("reference_embeddings:", reference_embeddings)

# ============================================================================

# 这里改成你上传的干净低噪声的wav格式语音文件
wav_path = 'goat_01.wav'

audio, source_sr = librosa.load(wav_path, sr=24000)
audio = audio / np.max(np.abs(audio))
audio.dtype = np.float32

start = time.time()
source = preprocess(audio)
keys = []
converted_samples = {}
reconstructed_samples = {}
converted_mels = {}

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
            mel = preprocess(wave)
            c = mel.transpose([0, 2, 1]).squeeze()
            recon = vocoder.inference(c)
            recon = recon.reshape([-1]).numpy()

    converted_samples[key] = y_out.numpy()
    reconstructed_samples[key] = recon
    converted_mels[key] = out
    keys.append(key)

end = time.time()

print('总共花费时间: %.3f sec' % (end - start))

print('原始语音 (使用声码器解码):')
wave, sr = librosa.load(wav_path, sr=24000)
mel = preprocess(wave)
c = mel.transpose([0, 2, 1]).squeeze()
with paddle.no_grad():
    recon = vocoder.inference(c)
    recon = recon.reshape([-1]).numpy()
# display(ipd.Audio(recon, rate=24000))
sf.write('orig_voc.wav', recon, samplerate=24000)

for key, wave in converted_samples.items():
    wav_name = 'vc_result_' + key + '.wav'
    print('语音转换结果: %s' % wav_name)
    sf.write(wav_name, wave, samplerate=24000)
    ref_wav_name = 'ref_voc_' + key + '.wav'
    print('参考的说话人 (使用声码器解码): %s' % ref_wav_name)
    if reconstructed_samples[key] is not None:
        sf.write(ref_wav_name, reconstructed_samples[key], samplerate=24000)
