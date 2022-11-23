# Parakeet
Parakeet aims to provide a flexible, efficient, and state-of-the-art text-to-speech toolkit for the open-source community. It is built on PaddlePaddle dynamic graph and includes many influential TTS models.  

<div align="center">
  <img src="../../images/logo.png" width=300 /> <br>
</div>

## Overview

To facilitate exploiting the existing TTS models directly and developing the new ones, Parakeet selects typical models and provides their reference implementations in PaddlePaddle. Furthermore, Parakeet abstracts the TTS pipeline and standardizes the procedure of data preprocessing, common modules sharing, model configuration, and the process of training and synthesis. The models supported here include Text FrontEnd, end-to-end Acoustic models, and Vocoders:

- Text FrontEnd
  - Rule-based Chinese frontend.

- Acoustic Models
  - [【FastSpeech2】FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
  - [【SpeedySpeech】SpeedySpeech: Efficient Neural Speech Synthesis](https://arxiv.org/abs/2008.03802)
  - [【Transformer TTS】Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)
  - [【Tacotron2】Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
- Vocoders
  - [【Parallel WaveGAN】Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480)
  - [【WaveFlow】WaveFlow: A Compact Flow-based Model for Raw Audio](https://arxiv.org/abs/1912.01219)
- Voice Cloning
  - [Transfer Learning from Speaker Verification to Multispeaker Text-to-Speech Synthesis](https://arxiv.org/pdf/1806.04558v4.pdf)
  - [【GE2E】Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)



## Audio samples

Check our [website](https://paddlespeech.readthedocs.io/en/latest/tts/demo.html) for audio sampels.

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
