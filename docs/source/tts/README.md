# Parakeet
Parakeet aims to provide a flexible, efficient and state-of-the-art text-to-speech toolkit for the open-source community. It is built on PaddlePaddle dynamic graph and includes many influential TTS models.  

<div align="center">
  <img src="docs/images/logo.png" width=300 /> <br>
</div>

## News  <img src="./docs/images/news_icon.png" width="40"/>
- Oct-12-2021, Refector examples code.
- Oct-12-2021, Parallel WaveGAN with LJSpeech. Check [examples/GANVocoder/parallelwave_gan/ljspeech](./examples/GANVocoder/parallelwave_gan/ljspeech).
- Oct-12-2021, FastSpeech2/FastPitch with LJSpeech. Check [examples/fastspeech2/ljspeech](./examples/fastspeech2/ljspeech).
- Sep-14-2021, Reconstruction of TransformerTTS. Check [examples/transformer_tts/ljspeech](./examples/transformer_tts/ljspeech).
- Aug-31-2021, Chinese Text Frontend. Check [examples/text_frontend](./examples/text_frontend).
- Aug-23-2021, FastSpeech2/FastPitch with AISHELL-3. Check [examples/fastspeech2/aishell3](./examples/fastspeech2/aishell3).
- Aug-03-2021, FastSpeech2/FastPitch with CSMSC. Check [examples/fastspeech2/baker](./examples/fastspeech2/baker).
- Jul-19-2021, SpeedySpeech with CSMSC. Check [examples/speedyspeech/baker](./examples/speedyspeech/baker).
- Jul-01-2021, Parallel WaveGAN with CSMSC. Check [examples/GANVocoder/parallelwave_gan/baker](./examples/GANVocoder/parallelwave_gan/baker).
- Jul-01-2021, Montreal-Forced-Aligner. Check  [examples/use_mfa](./examples/use_mfa).
- May-07-2021, Voice Cloning in Chinese. Check [examples/tacotron2_aishell3](./examples/tacotron2_aishell3).

## Overview

In order to facilitate exploiting the existing TTS models directly and developing the new ones, Parakeet selects typical models and provides their reference implementations in PaddlePaddle. Further more, Parakeet abstracts the TTS pipeline and standardizes the procedure of data preprocessing, common modules sharing, model configuration, and the process of training and synthesis. The models supported here include Text FrontEnd, end-to-end Acoustic models and Vocoders:

- Text FrontEnd
  - Rule based Chinese frontend.

- Acoustic Models
  - [【FastSpeech2】FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
  - [【SpeedySpeech】SpeedySpeech: Efficient Neural Speech Synthesis](https://arxiv.org/abs/2008.03802)
  - [【Transformer TTS】Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)
  - [【Tacotron2】Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
- Vocoders
  - [【Parallel WaveGAN】Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480)
  - [【WaveFlow】WaveFlow: A Compact Flow-based Model for Raw Audio](https://arxiv.org/abs/1912.01219)
- Voice Cloning
  - [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558v4.pdf)
  - [【GE2E】Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)

## Setup
It's difficult to install some dependent libraries for this repo in Windows system, we recommend that you **DO NOT** use Windows system, please use `Linux`.

Make sure the library `libsndfile1` is installed, e.g., on Ubuntu.

```bash
sudo apt-get install libsndfile1
```
### Install PaddlePaddle
See [install](https://www.paddlepaddle.org.cn/install/quick) for more details. This repo requires PaddlePaddle **2.1.2** or above.

### Install Parakeet

```bash
git clone https://github.com/PaddlePaddle/Parakeet
cd Parakeet
pip install -e .
```

If some python dependent packages cannot be installed successfully, you can run the following script first.
(replace `python3.6` with your own python version)
```bash
sudo apt install -y python3.6-dev
```

See [install](https://paddle-parakeet.readthedocs.io/en/latest/install.html) for more details.

## Examples
Entries to the introduction, and the launch of training and synthsis for different example models:

- [>>> Chinese Text Frontend](./examples/text_frontend)
- [>>> FastSpeech2/FastPitch](./examples/fastspeech2)
- [>>> Montreal-Forced-Aligner](./examples/use_mfa)
- [>>> Parallel WaveGAN](./examples/GANVocoder/parallelwave_gan)
- [>>> SpeedySpeech](./examples/speedyspeech)
- [>>> Tacotron2_AISHELL3](./examples/tacotron2_aishell3)
- [>>> GE2E](./examples/ge2e)
- [>>> WaveFlow](./examples/waveflow)
- [>>> TransformerTTS](./examples/transformer_tts)
- [>>> Tacotron2](./examples/tacotron2)

## Audio samples
### TTS models (Acoustic Model + Neural Vocoder)
Check our [website](https://paddleparakeet.readthedocs.io/en/latest/demo.html) for audio sampels.

## Released Model

### Acoustic Model

#### FastSpeech2/FastPitch
1. [fastspeech2_nosil_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_baker_ckpt_0.4.zip)
2. [fastspeech2_nosil_aishell3_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_aishell3_ckpt_0.4.zip)
3. [fastspeech2_nosil_ljspeech_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_ljspeech_ckpt_0.5.zip)

#### SpeedySpeech
1. [speedyspeech_nosil_baker_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/speedyspeech_nosil_baker_ckpt_0.5.zip)

#### TransformerTTS

1. [transformer_tts_ljspeech_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_ckpt_0.4.zip)

#### Tacotron2

1. [tacotron2_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_ckpt_0.3.zip)
2. [tacotron2_ljspeech_ckpt_0.3_alternative.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_ckpt_0.3_alternative.zip)

### Vocoder

#### WaveFlow

1. [waveflow_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_ljspeech_ckpt_0.3.zip)

#### Parallel WaveGAN

1. [pwg_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/pwg_baker_ckpt_0.4.zip)
2. [pwg_ljspeech_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/pwg_ljspeech_ckpt_0.5.zip)

### Voice Cloning

#### Tacotron2_AISHELL3

1. [tacotron2_aishell3_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_aishell3_ckpt_0.3.zip)

#### GE2E

1. [ge2e_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/ge2e_ckpt_0.3.zip)

## License

Parakeet is provided under the [Apache-2.0 license](LICENSE).
