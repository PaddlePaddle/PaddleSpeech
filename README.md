English | [简体中文](README_ch.md)



<p align="center">
  <img src="./docs/images/PaddleSpeech_log.png" />
</p>
<div align="center">  

  <h3>
  <a href="#quick-start"> Quick Start </a>
  | <a href="#tutorials"> Tutorials </a>
  | <a href="#models-list"> Models List </a>
</div>

------------------------------------------------------------------------------------
![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)

<!---
from https://github.com/18F/open-source-guide/blob/18f-pages/pages/making-readmes-readable.md
1.What is this repo or project? (You can reuse the repo description you used earlier because this section doesn’t have to be long.)
2.How does it work?
3.Who will use this repo or project?
4.What is the goal of this project?
-->

**PaddleSpeech** is an open-source toolkit on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) platform for a variety of critical tasks in speech, with the state-of-art and influential models.

##### Speech-To-Text

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

##### Text-To-Speech
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

For more synthesized audios, please refer to [PaddleSpeech Text-To-Speech samples](https://paddlespeech.readthedocs.io/en/latest/tts/demo.html).

Via the easy-to-use, efficient, flexible and scalable implementation, our vision is to empower both industrial application and academic research, including training, inference & testing modules, and deployment process. To be more specific, this toolkit features at:
- **Fast and Light-weight**: we provide high-speed and ultra-lightweight models that are convenient for industrial deployment.
- **Rule-based Chinese frontend**: our frontend contains Text Normalization and Grapheme-to-Phoneme (G2P, including Polyphone and Tone Sandhi). Moreover, we use self-defined linguistic rules to adapt Chinese context.
- **Varieties of Functions that Vitalize both Industrial and Academia**:
  - *Implementation of critical audio tasks*: this toolkit contains audio functions like Speech Translation, Automatic Speech Recognition, Text-To-Speech Synthesis, Voice Cloning, etc.
  - *Integration of mainstream models and datasets*: the toolkit implements modules that participate in the whole pipeline of the speech tasks, and uses mainstream datasets like LibriSpeech, LJSpeech, AIShell, CSMSC, etc. See also [model lists](#models-list) for more details.
  - *Cascaded models application*: as an extension of the application of traditional audio tasks, we combine the workflows of aforementioned tasks with other fields like Natural language processing (NLP), like Punctuation Restoration.

# Community

You are warmly welcome to submit questions in [discussions](https://github.com/PaddlePaddle/DeepSpeech/discussions) and bug reports in [issues](https://github.com/PaddlePaddle/DeepSpeech/issues)! Also, we highly appreciate if you would like to contribute to this project!

If you are from China, we strongly recommend you join our PaddleSpeech WeChat group. Scan the following WeChat QR code and get in touch with the other developers in this community!

<div align="center">
<img src="./docs/images/wechat-code-speech.png"  width = "200">

</div>

# Alternative Installation

The base environment in this page is  
- Ubuntu 16.04
- python>=3.7
- paddlepaddle==2.1.2

If you want to set up PaddleSpeech in other environment, please see the [installation](./docs/source/install.md) documents for all the alternatives.

# Quick Start

Developers can have a try of our model with only a few lines of code.

A tiny DeepSpeech2 **Speech-To-Text** model training on toy set of LibriSpeech:

```shell
cd examples/tiny/s0/
# source the environment
source path.sh
# prepare librispeech dataset
bash local/data.sh
# evaluate your ckptfile model file
bash local/test.sh conf/deepspeech2.yaml ckptfile offline
```

For **Text-To-Speech**, try pretrained FastSpeech2 + Parallel WaveGAN on CSMSC:
```shell
cd examples/csmsc/tts3
# download the pretrained models and unaip them
wget https://paddlespeech.bj.bcebos.com/Parakeet/pwg_baker_ckpt_0.4.zip
unzip pwg_baker_ckpt_0.4.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_baker_ckpt_0.4.zip
unzip fastspeech2_nosil_baker_ckpt_0.4.zip
# source the environment
source path.sh
# run end-to-end synthesize
FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/synthesize_e2e.py \
  --fastspeech2-config=fastspeech2_nosil_baker_ckpt_0.4/default.yaml \
  --fastspeech2-checkpoint=fastspeech2_nosil_baker_ckpt_0.4/snapshot_iter_76000.pdz \
  --fastspeech2-stat=fastspeech2_nosil_baker_ckpt_0.4/speech_stats.npy \
  --pwg-config=pwg_baker_ckpt_0.4/pwg_default.yaml \
  --pwg-checkpoint=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
  --pwg-stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
  --text=${BIN_DIR}/../sentences.txt \
  --output-dir=exp/default/test_e2e \
  --inference-dir=exp/default/inference \
  --device="gpu" \
  --phones-dict=fastspeech2_nosil_baker_ckpt_0.4/phone_id_map.txt
```

If you want to try more functions like training and tuning, please see [Speech-To-Text Quick Start](./docs/source/asr/quick_start.md) and [Text-To-Speech Quick Start](./docs/source/tts/quick_start.md).

# Models List

PaddleSpeech supports a series of most popular models, summarized in [released models](./docs/source/released_models.md) with available pretrained models.

Speech-To-Text module contains *Acoustic Model* and *Language Model*, with the following details:

<!---
The current hyperlinks redirect to [Previous Parakeet](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples).
-->

<table style="width:100%">
  <thead>
    <tr>
      <th>Speech-To-Text Module Type</th>
      <th>Dataset</th>
      <th>Model Type</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Acoustic Model</td>
      <td rowspan="2" >Aishell</td>
      <td >DeepSpeech2 RNN + Conv based Models</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/release2.1/aishell/s0/aishell.s0.ds_online.5rnn.debug.tar.gz">Online</a> / <a href = "https://deepspeech.bj.bcebos.com/release2.1/aishell/s0/aishell.s0.ds2.offline.cer6p65.release.tar.gz">Offline </a>
      </td>
    </tr>
    <tr>
      <td>Transformer based Attention Models </td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/release2.1/librispeech/s1/conformer.release.tar.gz">Non-CTC Loss</a> / <a href = "https://deepspeech.bj.bcebos.com/release2.1/aishell/s1/aishell.release.tar.gz">CTC Loss </a>
      </td>
    </tr>
      <tr>
      <td> Librispeech</td>
      <td>Transformer based Attention Models </td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/release2.1/librispeech/s1/conformer.release.tar.gz">Conformer</a> / <a href = "https://deepspeech.bj.bcebos.com/release2.1/aishell/s1/aishell.release.tar.gz"> Transformer </a> Decoder
      </td>
      </td>
    </tr>
   <tr>
      <td rowspan="2">Language Model</td>
      <td >CommonCrawl(en.00)</td>
      <td >English Language Model</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm">English Language Model</a>
      </td>
    </tr>
    <tr>
      <td>Baidu Internal Corpus</td>
      <td>Mandarin Language Model Small</td>
      <td>
      Mandarin Language Model <a href = "https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm"> Small</a> / <a href = "https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm">Large</a>
      </td>
    </tr>
  </tbody>
</table>


PaddleSpeech Text-To-Speech mainly contains three modules: *Text Frontend*, *Acoustic Model* and *Vocoder*. Acoustic Model and Vocoder models are listed as follow:

<table>
  <thead>
    <tr>
      <th> Text-To-Speech Module Type <img width="110" height="1"> </th>
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
      <td>TransformerTTS</td>
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
      <td rowspan="2">Vocoder</td>
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
    <td rowspan="2">Voice Cloning</td>
    <td>GE2E</td>
    <td >AISHELL-3, etc.</td>
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
    </td>
    </tr>
  </tbody>
</table>

# Tutorials

Normally, [Speech SoTA](https://paperswithcode.com/area/speech) gives you an overview of the hot academic topics in speech. To focus on the tasks in PaddleSpeech, you will find the following guidelines are helpful to grasp the core ideas.
- [Overview](./docs/source/introduction.md)
- Quick Start
  - [Dependencies](./docs/source/dependencies.md) and [Installation](./docs/source/install.md)
  - [Quick Start of Speech-To-Text](./docs/source/asr/quick_start.md)
  - [Quick Start of Text-To-Speech](./docs/source/tts/quick_start.md)
- Speech-To-Text
  - [Models Introduction](./docs/source/asr/models_introduction.md)
  - [Data Preparation](./docs/source/asr/data_preparation.md)
  - [Data Augmentation Pipeline](./docs/source/asr/augmentation.md)
  - [Features](./docs/source/asr/feature_list.md)
  - [Ngram LM](./docs/source/asr/ngram_lm.md)
- Text-To-Speech
  - [Introduction](./docs/source/tts/models_introduction.md)
  - [Advanced Usage](./docs/source/tts/advanced_usage.md)
  - [Chinese Rule Based Text Frontend](./docs/source/tts/zh_text_frontend.md)
  - [Test Audio Samples](https://paddlespeech.readthedocs.io/en/latest/tts/demo.html) and [PaddleSpeech VS. Espnet](https://paddlespeech.readthedocs.io/en/latest/tts/demo_2.html)
- [Released Models](./docs/source/released_model.md)

The TTS module is originally called [Parakeet](https://github.com/PaddlePaddle/Parakeet), and now merged with DeepSpeech. If you are interested in academic research about this function, please see [TTS research overview](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/docs/source/tts#overview). Also, [this document](https://paddleparakeet.readthedocs.io/en/latest/released_models.html) is a good guideline for the pipeline components.

## FAQ and Contributing
You are warmly welcome to submit questions in [discussions](https://github.com/PaddlePaddle/DeepSpeech/discussions) and bug reports in [issues](https://github.com/PaddlePaddle/DeepSpeech/issues)! Also, we highly appreciate if you would like to contribute to this project!

# License and Acknowledgement
PaddleSpeech is provided under the [Apache-2.0 License](./LICENSE).

PaddleSpeech depends on a lot of open source repositories. See [references](./docs/source/reference.md) for more information.

# Citation
To cite PaddleSpeech for research, please use the following format.
```tex
@misc{ppspeech2021,
title={PaddleSpeech, a toolkit for audio processing based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/DeepSpeech}},
year={2021}
}
```
