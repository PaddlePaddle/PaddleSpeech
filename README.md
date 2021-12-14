<p align="center">
  <img src="./docs/images/PaddleSpeech_logo.png" />
</p>
<div align="center">  

  <h3>
  <a href="#quick-start"> Quick Start </a>
  | <a href="#tutorials"> Tutorials </a>
  | <a href="#model-list"> Models List </a>
</div>

------------------------------------------------------------------------------------

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)
![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)

<!---
from https://github.com/18F/open-source-guide/blob/18f-pages/pages/making-readmes-readable.md
1.What is this repo or project? (You can reuse the repo description you used earlier because this section doesn’t have to be long.)
2.How does it work?
3.Who will use this repo or project?
4.What is the goal of this project?
-->


**PaddleSpeech** is an open-source toolkit on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) platform for a variety of critical tasks in speech and audio, with the state-of-art and influential models.

##### Speech-to-Text

<div align = "center">
<table style="width:100%">
  <thead>
    <tr>
      <th> Input Audio  </th>
      <th width="550"> Recognition Result  </th>
    </tr>
  </thead>
  <tbody>
   <tr>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200 style="max-width: 100%;"></a><br>
      </td>
      <td >I knocked at the door on the ancient side of the building.</td>
    </tr>
    <tr>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
      <td>我认为跑步最重要的就是给我带来了身体健康。</td>
    </tr>
    
  </tbody>
</table>

</div>

##### Speech Translation (English to Chinese)

<div align = "center">
<table style="width:100%">
  <thead>
    <tr>
      <th> Input Audio  </th>
      <th width="550"> Translations Result  </th>
    </tr>
  </thead>
  <tbody>
   <tr>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200 style="max-width: 100%;"></a><br>
      </td>
      <td >我 在 这栋 建筑 的 古老 门上 敲门。</td>
    </tr>
  </tbody>
</table>

</div>

##### Text-to-Speech
<div align = "center">
<table style="width:100%">
  <thead>
    <tr>
      <th><img width="200" height="1"> Input Text <img width="200" height="1"> </th>
      <th>Synthetic Audio</th>
    </tr>
  </thead>
  <tbody>
   <tr>
      <td >Life was like a box of chocolates, you never know what you're gonna get.</td>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/001.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
    <tr>
      <td >早上好，今天是2020/10/29，最低温度是-3°C。</td>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/parakeet_espnet_fs2_pwg_demo/tn_g2p/parakeet/001.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
  </tbody>
</table>

</div>

For more synthesized audios, please refer to [PaddleSpeech Text-to-Speech samples](https://paddlespeech.readthedocs.io/en/latest/tts/demo.html).

### Features:

Via the easy-to-use, efficient, flexible and scalable implementation, our vision is to empower both industrial application and academic research, including training, inference & testing modules, and deployment process. To be more specific, this toolkit features at:
- 📦 **Ease of Use**: low barriers to install, and [CLI](#quick-start) is available to quick-start your journey.
- 🏆 **Align to the State-of-the-Art**: we provide high-speed and ultra-lightweight models, and also cutting-edge technology. 
- 💯 **Rule-based Chinese frontend**: our frontend contains Text Normalization and Grapheme-to-Phoneme (G2P, including Polyphone and Tone Sandhi). Moreover, we use self-defined linguistic rules to adapt Chinese context.
- **Varieties of Functions that Vitalize both Industrial and Academia**:
  - 🛎️ *Implementation of critical audio tasks*: this toolkit contains audio functions like  Audio Classification, Speech Translation, Automatic Speech Recognition, Text-to-Speech Synthesis, etc.
  - 🔬 *Integration of mainstream models and datasets*: the toolkit implements modules that participate in the whole pipeline of the speech tasks, and uses mainstream datasets like LibriSpeech, LJSpeech, AIShell, CSMSC, etc. See also [model list](#model-list) for more details.
  - 🧩 *Cascaded models application*: as an extension of the typical traditional audio tasks, we combine the workflows of the aforementioned tasks with other fields like Natural language processing (NLP) and Computer Vision (CV).


### Recent Update:

<!---
2021.12.14: We would like to have an online courses to introduce basics and research of speech, as well as code practice with `paddlespeech`. Please pay attention to our [Calendar](https://www.paddlepaddle.org.cn/live).
--->
- 🤗 2021.12.14: Our PaddleSpeech [ASR](https://huggingface.co/spaces/KPatrick/PaddleSpeechASR) and [TTS](https://huggingface.co/spaces/akhaliq/paddlespeech) Demos on Hugging Face Spaces are available!
- 👏🏻 2021.12.10: PaddleSpeech CLI is available for Audio Classification, Automatic Speech Recognition, Speech Translation (English to Chinese) and Text-to-Speech.

### Communication
If you are in China, we recommend you to join our WeChat group to contact directly with our team members!

<div align="center">
<img src="./docs/images/wechat_group.png"  width = "200"  />
</div>

## Installation

We strongly recommend our users to install PaddleSpeech in **Linux** with *python>=3.7*, where `paddlespeech`  can be easily installed with `pip`:
```python
pip install paddlepaddle paddlespeech
```
Up to now, **Mac OSX** supports CLI for the all our tasks, Windows only supports PaddleSpeech CLI for Audio Classification, Speech-to-Text and Text-to-Speech. Please see [installation](./docs/source/install.md) for other alternatives.

## Quick Start

Developers can have a try of our models with [PaddleSpeech Command Line](./paddlespeech/cli/README.md). Change `--input` to test your own audio/text.

**Audio Classification**     
```shell
paddlespeech cls --input input.wav
```
**Automatic Speech Recognition**
```shell
paddlespeech asr --lang zh --input input_16k.wav
```
**Speech Translation** (English to Chinese)
```shell
paddlespeech st --input input_16k.wav
```
**Text-to-Speech** 
```shell
paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav
```

If you want to try more functions like training and tuning, please have a look at documents of [Speech-to-Text](./docs/source/asr/quick_start.md) and [Text-to-Speech](./docs/source/tts/quick_start.md).

## Model List

PaddleSpeech supports a series of most popular models. They are summarized in [released models](./docs/source/released_model.md) and attached with available pretrained models.

**Speech-to-Text** contains *Acoustic Model* and *Language Model*, with the following details:

<!---
The current hyperlinks redirect to [Previous Parakeet](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples).
-->

<table style="width:100%">
  <thead>
    <tr>
      <th>Speech-to-Text Module Type</th>
      <th>Dataset</th>
      <th>Model Type</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Speech Recogination</td>
      <td rowspan="2" >Aishell</td>
      <td >DeepSpeech2 RNN + Conv based Models</td>
      <td>
      <a href = "./examples/aishell/asr0">deepspeech2-aishell</a>
      </td>
    </tr>
    <tr>
      <td>Transformer based Attention Models </td>
      <td>
      <a href = "./examples/aishell/asr1">u2.transformer.conformer-aishell</a>
      </td>
    </tr>
      <tr>
      <td> Librispeech</td>
      <td>Transformer based Attention Models </td>
      <td>
      <a href = "./examples/librispeech/asr0">deepspeech2-librispeech</a> / <a href = "./examples/librispeech/asr1">transformer.conformer.u2-librispeech</a>  / <a href = "./examples/librispeech/asr2">transformer.conformer.u2-kaldi-librispeech</a>
      </td>
      </td>
    </tr>
  <tr>
  <td>Alignment</td>
  <td>THCHS30</td>
  <td>MFA</td>
  <td>
  <a href = ".examples/thchs30/align0">mfa-thchs30</a>
  </td>
  </tr>
   <tr>
      <td rowspan="2">Language Model</td>
      <td colspan = "2">Ngram Language Model</td>
      <td>
      <a href = "./examples/other/ngram_lm">kenlm</a>
      </td>
    </tr>
    <tr>
      <td>TIMIT</td>
      <td>Unified Streaming & Non-streaming Two-pass</td>
      <td>
    <a href = "./examples/timit/asr1"> u2-timit</a>
      </td>
    </tr>
    <tr>
      <td rowspan="2">Speech Translation (English to Chinese)</td> 
      <td rowspan="2">TED En-Zh</td>
      <td>Transformer + ASR MTL</td>
      <td>
      <a href = "./examples/ted_en_zh/st0">transformer-ted</a>
      </td>
  </tr>
  <tr>
      <td>FAT + Transformer + ASR MTL</td>
      <td>
      <a href = "./examples/ted_en_zh/st1">fat-st-ted</a>
      </td>
  </tr>
  </tbody>
</table>

**Text-to-Speech** in PaddleSpeech mainly contains three modules: *Text Frontend*, *Acoustic Model* and *Vocoder*. Acoustic Model and Vocoder models are listed as follow:

<table>
  <thead>
    <tr>
      <th> Text-to-Speech Module Type <img width="110" height="1"> </th>
      <th>  Model Type  </th>
      <th> <img width="50" height="1"> Dataset  <img width="50" height="1"> </th>
      <th> <img width="101" height="1"> Link <img width="105" height="1"> </th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td> Text Frontend</td>
    <td colspan="2"> &emsp; </td>
    <td>
    <a href = "./examples/other/tn">tn</a> / <a href = "./examples/other/g2p">g2p</a>
    </td>
    </tr>
    <tr>
      <td rowspan="4">Acoustic Model</td>
      <td >Tacotron2</td>
      <td rowspan="2" >LJSpeech</td>
      <td>
      <a href = "./examples/ljspeech/tts0">tacotron2-ljspeech</a>
      </td>
    </tr>
    <tr>
      <td>Transformer TTS</td>
      <td>
      <a href = "./examples/ljspeech/tts1">transformer-ljspeech</a>
      </td>
    </tr>
    <tr>
      <td>SpeedySpeech</td>
      <td>CSMSC</td>
      <td >
      <a href = "./examples/csmsc/tts2">speedyspeech-csmsc</a>
      </td>
    </tr>
    <tr>
      <td>FastSpeech2</td>
      <td>AISHELL-3 / VCTK / LJSpeech / CSMSC</td>
      <td>
      <a href = "./examples/aishell3/tts3">fastspeech2-aishell3</a> / <a href = "./examples/vctk/tts3">fastspeech2-vctk</a> / <a href = "./examples/ljspeech/tts3">fastspeech2-ljspeech</a> / <a href = "./examples/csmsc/tts3">fastspeech2-csmsc</a>
      </td>
    </tr>
   <tr>
      <td rowspan="3">Vocoder</td>
      <td >WaveFlow</td>
      <td >LJSpeech</td>
      <td>
      <a href = "./examples/ljspeech/voc0">waveflow-ljspeech</a>
      </td>
    </tr>
    <tr>
      <td >Parallel WaveGAN</td>
      <td >LJSpeech / VCTK / CSMSC</td>
      <td>
      <a href = "./examples/ljspeech/voc1">PWGAN-ljspeech</a> / <a href = "./examples/vctk/voc1">PWGAN-vctk</a> / <a href = "./examples/csmsc/voc1">PWGAN-csmsc</a>
      </td>
    </tr>
    <tr>
      <td >Multi Band MelGAN</td>
      <td >CSMSC</td>
      <td>
      <a href = "./examples/csmsc/voc3">Multi Band MelGAN-csmsc</a> 
      </td>
    </tr>                                                                                                                                           
    <tr>
      <td rowspan="3">Voice Cloning</td>
      <td>GE2E</td>
      <td >Librispeech, etc.</td>
      <td>
      <a href = "./examples/other/ge2e">ge2e</a>
      </td>
    </tr>
    <tr>
      <td>GE2E + Tactron2</td>
      <td>AISHELL-3</td>
      <td>
      <a href = "./examples/aishell3/vc0">ge2e-tactron2-aishell3</a>
      </td>
    </tr>
    <tr>
      <td>GE2E + FastSpeech2</td>
      <td>AISHELL-3</td>
      <td>
      <a href = "./examples/aishell3/vc1">ge2e-fastspeech2-aishell3</a>
      </td>
    </tr>
  </tbody>
</table>

**Audio Classification**

<table style="width:100%">
  <thead>
    <tr>
      <th> <img width="150" height="1">Task <img width="150" height="1"></th>
      <th> <img width="110" height="1">Dataset <img width="110" height="1"></th>
      <th> <img width="110" height="1">Model Type <img width="110" height="1"></th>
      <th> <img width="110" height="1">Link <img width="110" height="1"></th>
    </tr>
  </thead>
  <tbody>
  

  <tr>
      <td>Audio Classification</td>
      <td>ESC-50</td>
      <td>PANN</td>
      <td>
      <a href = "./examples/esc50/cls0">pann-esc50</a>
      </td>
    </tr>
  </tbody>
</table>

## Documents

Normally, [Speech SoTA](https://paperswithcode.com/area/speech), [Audio SoTA](https://paperswithcode.com/area/audio) and [Music SoTA](https://paperswithcode.com/area/music) give you an overview of the hot academic topics in the related area. To focus on the tasks in PaddleSpeech, you will find the following guidelines are helpful to grasp the core ideas.

- [Installation](./docs/source/install.md)
- Tutorials
  - [Automatic Speech Recognition](./docs/source/asr/quick_start.md)
    - [Introduction](./docs/source/asr/models_introduction.md)
    - [Data Preparation](./docs/source/asr/data_preparation.md)
    - [Data Augmentation](./docs/source/asr/augmentation.md)
    - [Ngram LM](./docs/source/asr/ngram_lm.md)
  - [Text-to-Speech](./docs/source/tts/quick_start.md)
    - [Introduction](./docs/source/tts/models_introduction.md)
    - [Advanced Usage](./docs/source/tts/advanced_usage.md)
    - [Chinese Rule Based Text Frontend](./docs/source/tts/zh_text_frontend.md)
    - [Test Audio Samples](https://paddlespeech.readthedocs.io/en/latest/tts/demo.html)
  - Audio Classification
  - Speech Translation
- [Released Models](./docs/source/released_model.md)

The Text-to-Speech module is originally called [Parakeet](https://github.com/PaddlePaddle/Parakeet), and now merged with this repository. If you are interested in academic research about this task, please see [TTS research overview](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/docs/source/tts#overview). Also, [this document](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/tts/models_introduction.md) is a good guideline for the pipeline components.

## Citation

To cite PaddleSpeech for research, please use the following format.
```tex
@misc{ppspeech2021,
title={PaddleSpeech, a toolkit for audio processing based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleSpeech}},
year={2021}
}
```

## Contribute to PaddleSpeech

You are warmly welcome to submit questions in [discussions](https://github.com/PaddlePaddle/PaddleSpeech/discussions) and bug reports in [issues](https://github.com/PaddlePaddle/PaddleSpeech/issues)! Also, we highly appreciate if you are willing to contribute to this project!

### Contributors
<p align="center">
<a href="https://github.com/zh794390558"><img src="https://avatars.githubusercontent.com/u/3038472?v=4" width=75 height=75></a>
<a href="https://github.com/Jackwaterveg"><img src="https://avatars.githubusercontent.com/u/87408988?v=4" width=75 height=75></a>
<a href="https://github.com/yt605155624"><img src="https://avatars.githubusercontent.com/u/24568452?v=4" width=75 height=75></a>
<a href="https://github.com/kuke"><img src="https://avatars.githubusercontent.com/u/3064195?v=4" width=75 height=75></a>
<a href="https://github.com/xinghai-sun"><img src="https://avatars.githubusercontent.com/u/7038341?v=4" width=75 height=75></a>
<a href="https://github.com/pkuyym"><img src="https://avatars.githubusercontent.com/u/5782283?v=4" width=75 height=75></a>
<a href="https://github.com/KPatr1ck"><img src="https://avatars.githubusercontent.com/u/22954146?v=4" width=75 height=75></a>
<a href="https://github.com/LittleChenCc"><img src="https://avatars.githubusercontent.com/u/10339970?v=4" width=75 height=75></a>
<a href="https://github.com/745165806"><img src="https://avatars.githubusercontent.com/u/20623194?v=4" width=75 height=75></a>
<a href="https://github.com/Mingxue-Xu"><img src="https://avatars.githubusercontent.com/u/92848346?v=4" width=75 height=75></a>
<a href="https://github.com/chrisxu2016"><img src="https://avatars.githubusercontent.com/u/18379485?v=4" width=75 height=75></a>
<a href="https://github.com/lfchener"><img src="https://avatars.githubusercontent.com/u/6771821?v=4" width=75 height=75></a>
<a href="https://github.com/luotao1"><img src="https://avatars.githubusercontent.com/u/6836917?v=4" width=75 height=75></a>
<a href="https://github.com/wanghaoshuang"><img src="https://avatars.githubusercontent.com/u/7534971?v=4" width=75 height=75></a>
<a href="https://github.com/gongel"><img src="https://avatars.githubusercontent.com/u/24390500?v=4" width=75 height=75></a>
<a href="https://github.com/mmglove"><img src="https://avatars.githubusercontent.com/u/38800877?v=4" width=75 height=75></a>
<a href="https://github.com/iclementine"><img src="https://avatars.githubusercontent.com/u/16222986?v=4" width=75 height=75></a>
<a href="https://github.com/ZeyuChen"><img src="https://avatars.githubusercontent.com/u/1371212?v=4" width=75 height=75></a>
<a href="https://github.com/AK391"><img src="https://avatars.githubusercontent.com/u/81195143?v=4" width=75 height=75></a>
<a href="https://github.com/qingqing01"><img src="https://avatars.githubusercontent.com/u/7845005?v=4" width=75 height=75></a>
<a href="https://github.com/ericxk"><img src="https://avatars.githubusercontent.com/u/4719594?v=4" width=75 height=75></a>
<a href="https://github.com/kvinwang"><img src="https://avatars.githubusercontent.com/u/6442159?v=4" width=75 height=75></a>
<a href="https://github.com/jiqiren11"><img src="https://avatars.githubusercontent.com/u/82639260?v=4" width=75 height=75></a>
<a href="https://github.com/AshishKarel"><img src="https://avatars.githubusercontent.com/u/58069375?v=4" width=75 height=75></a>
<a href="https://github.com/chesterkuo"><img src="https://avatars.githubusercontent.com/u/6285069?v=4" width=75 height=75></a>
<a href="https://github.com/tensor-tang"><img src="https://avatars.githubusercontent.com/u/21351065?v=4" width=75 height=75></a>
<a href="https://github.com/hysunflower"><img src="https://avatars.githubusercontent.com/u/52739577?v=4" width=75 height=75></a>  
<a href="https://github.com/wwhu"><img src="https://avatars.githubusercontent.com/u/6081200?v=4" width=75 height=75></a>
<a href="https://github.com/lispc"><img src="https://avatars.githubusercontent.com/u/2833376?v=4" width=75 height=75></a>
<a href="https://github.com/jerryuhoo"><img src="https://avatars.githubusercontent.com/u/24245709?v=4" width=75 height=75></a>
<a href="https://github.com/harisankarh"><img src="https://avatars.githubusercontent.com/u/1307053?v=4" width=75 height=75></a>
<a href="https://github.com/Jackiexiao"><img src="https://avatars.githubusercontent.com/u/18050469?v=4" width=75 height=75></a>
<a href="https://github.com/limpidezza"><img src="https://avatars.githubusercontent.com/u/71760778?v=4" width=75 height=75></a>
</p>

## Acknowledgement

- Many thanks to [yeyupiaoling](https://github.com/yeyupiaoling) for years of attention, constructive advice and great help.
- Many thanks to [AK391](https://github.com/AK391) for TTS web demo on Huggingface Spaces using Gradio.

Besides, PaddleSpeech depends on a lot of open source repositories. See [references](./docs/source/reference.md) for more information.

## License

PaddleSpeech is provided under the [Apache-2.0 License](./LICENSE).
