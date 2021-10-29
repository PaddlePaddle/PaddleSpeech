English | [简体中文](README_ch.md)

# PaddleSpeech



<p align="center">
  <img src="./docs/images/PaddleSpeech_log.png" />
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

<!---
why they should use your module,
how they can install it,
how they can use it
-->

**PaddleSpeech** is an open-source toolkit on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) platform for a variety of critical tasks in speech, with state-of-art and influential models.

Via the easy-to-use, efficient, flexible and scalable implementation, our vision is to empower both industrial application and academic research, including training, inference & testing modules, and deployment process. To be more specific, this toolkit features at:
- **Fast and Light-weight**: we provide high-speed and ultra-lightweight models that are convenient for industrial deployment.
- **Rule-based Chinese frontend**: our frontend contains Text Normalization (TN) and Grapheme-to-Phoneme (G2P, including Polyphone and Tone Sandhi). Moreover, we use self-defined linguistic rules to adapt Chinese context.
- **Varieties of Functions that Vitalize both Industrial and Academia**:
  - *Implementation of critical audio tasks*: this toolkit contains audio functions like Speech Translation (ST), Automatic Speech Recognition (ASR), Text-To-Speech Synthesis (TTS), Voice Cloning(VC), Punctuation Restoration, etc.
  - *Integration of mainstream models and datasets*: the toolkit implements modules that participate in the whole pipeline of the speech tasks, and uses mainstream datasets like LibriSpeech, LJSpeech, AIShell, CSMSC, etc. See also [model lists](#models-list) for more details.
  - *Cross-domain application*: as an extension of the application of traditional audio tasks, we combine the aforementioned tasks with other fields like NLP.

Let's install PaddleSpeech with only a few lines of code!

>Note: The official name is still deepspeech. 2021/10/26

If you are using Ubuntu, PaddleSpeech can be set up with pip installation (with root privilege).
```shell
git clone https://github.com/PaddlePaddle/DeepSpeech.git
cd DeepSpeech
pip install -e .
```

## Table of Contents

The contents of this README is as follow:
- [Alternative Installation](#alternative-installation)
- [Quick Start](#quick-start)
- [Models List](#models-list)
- [Tutorials](#tutorials)
- [FAQ and Contributing](#faq-and-contributing)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Alternative Installation

The base environment in this page is  
- Ubuntu 16.04
- python>=3.7
- paddlepaddle==2.1.2

If you want to set up PaddleSpeech in other environment, please see the [ASR installation](docs/source/asr/install.md) and [TTS installation](docs/source/tts/install.md) documents for all the alternatives.

## Quick Start
> Note: the current links to `English ASR` and `English TTS` are not valid.

Just a quick test of our functions: [English ASR](link/hubdetail?name=deepspeech2_aishell&en_category=AutomaticSpeechRecognition) and [English TTS](link/hubdetail?name=fastspeech2_baker&en_category=TextToSpeech) by typing message or upload your own audio file.

Developers can have a try of our model with only a few lines of code.

A tiny *ASR* DeepSpeech2 model training on toy set of LibriSpeech:

```shell
cd examples/tiny/s0/
# source the environment
source path.sh
# prepare librispeech dataset
bash local/data.sh
# evaluate your ckptfile model file
bash local/test.sh conf/deepspeech2.yaml ckptfile offline
```

For *TTS*, try FastSpeech2 on LJSpeech:
- Download LJSpeech-1.1 from the [ljspeech official website](https://keithito.com/LJ-Speech-Dataset/), our prepared durations for fastspeech2 [ljspeech_alignment](https://paddlespeech.bj.bcebos.com/MFA/LJSpeech-1.1/ljspeech_alignment.tar.gz).
- The pretrained models are seperated into two parts: [fastspeech2_nosil_ljspeech_ckpt](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_ljspeech_ckpt_0.5.zip) and [pwg_ljspeech_ckpt](https://paddlespeech.bj.bcebos.com/Parakeet/pwg_ljspeech_ckpt_0.5.zip). Please download then unzip to `./model/fastspeech2` and `./model/pwg` respectively.
- Assume your path to the dataset is `~/datasets/LJSpeech-1.1` and `./ljspeech_alignment` accordingly, preprocess your data and then use our pretrained model to synthesize:
```shell
bash ./local/preprocess.sh conf/default.yaml
bash ./local/synthesize_e2e.sh conf/default.yaml ./model/fastspeech2/snapshot_iter_100000.pdz ./model/pwg/pwg_snapshot_iter_400000.pdz
```



If you want to try more functions like training and tuning, please see [ASR getting started](docs/source/asr/getting_started.md) and [TTS Basic Use](/docs/source/tts/basic_usage.md).

## Models List

PaddleSpeech supports a series of most popular models, summarized in [released models](./docs/source/released_model.md) with available pretrained models.

ASR module contains *Acoustic Model* and *Language Model*, with the following details:

<!---
The current hyperlinks redirect to [Previous Parakeet](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples).
-->

> Note: The `Link` should be code path rather than download links.


<table>
  <thead>
    <tr>
      <th>ASR Module Type</th>
      <th>Dataset</th>
      <th>Model Type</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Acoustic Model</td>
      <td rowspan="4" >Aishell</td>
      <td >2 Conv + 5 LSTM layers with only forward direction</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/release2.1/aishell/s0/aishell.s0.ds_online.5rnn.debug.tar.gz">Ds2 Online Aishell Model</a>
      </td>
    </tr>
    <tr>
      <td>2 Conv + 3 bidirectional GRU layers</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/release2.1/aishell/s0/aishell.s0.ds2.offline.cer6p65.release.tar.gz">Ds2 Offline Aishell Model</a>
      </td>
    </tr>
    <tr>
      <td>Encoder:Conformer, Decoder:Transformer, Decoding method: Attention + CTC</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/release2.1/aishell/s1/aishell.release.tar.gz">Conformer Offline Aishell Model</a>
      </td>
    </tr>
    <tr>
      <td >Encoder:Conformer, Decoder:Transformer, Decoding method: Attention</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/release2.1/librispeech/s1/conformer.release.tar.gz">Conformer Librispeech Model</a>
      </td>
    </tr>
      <tr>
      <td rowspan="2"> Librispeech</td>
      <td>Encoder:Conformer, Decoder:Transformer, Decoding method: Attention</td>
      <td> <a href = "https://deepspeech.bj.bcebos.com/release2.1/librispeech/s1/conformer.release.tar.gz">Conformer Librispeech Model</a> </td>
    </tr>
    <tr>
      <td>Encoder:Transformer, Decoder:Transformer, Decoding method: Attention</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/release2.1/librispeech/s1/transformer.release.tar.gz">Transformer Librispeech Model</a>
      </td>
    </tr>
   <tr>
      <td rowspan="3">Language Model</td>
      <td >CommonCrawl(en.00)</td>
      <td >English Language Model</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm">English Language Model</a>
      </td>
    </tr>
    <tr>
      <td rowspan="2">Baidu Internal Corpus</td>
      <td>Mandarin Language Model Small</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm">Mandarin Language Model Small</a>
      </td>
    </tr>
    <tr>
      <td >Mandarin Language Model Large</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm">Mandarin Language Model Large</a>
      </td>
    </tr>
  </tbody>
</table>


PaddleSpeech TTS mainly contains three modules: *Text Frontend*, *Acoustic Model* and *Vocoder*. Acoustic Model and Vocoder models are listed as follow:

<table>
  <thead>
    <tr>
      <th>TTS Module Type</th>
      <th>Model Type</th>
      <th>Dataset</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td> Text Frontend</td>
    <td colspan="2"> &emsp; </td>
    <td>
    <a href = "./examples/other/text_frontend">chinese-fronted</a>
    </td>
    </tr>
    <tr>
      <td rowspan="7">Acoustic Model</td>
      <td >Tacotron2</td>
      <td rowspan="2" >LJSpeech</td>
      <td>
      <a href = "./examples/ljspeech/tts0">tacotron2-vctk</a>
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
      <td rowspan="4">FastSpeech2</td>
      <td>AISHELL-3</td>
      <td>
      <a href = "./examples/aishell3/tts3">fastspeech2-aishell3</a>
      </td>
    </tr>
    <tr>
      <td>VCTK</td>
      <td> <a href = "./examples/vctk/tts3">fastspeech2-vctk</a> </td>
    </tr>
    <tr>
      <td>LJSpeech</td>
      <td> <a href = "./examples/ljspeech/tts3">fastspeech2-ljspeech</a> </td>
    </tr>
    <tr>
      <td>CSMSC</td>
      <td>
      <a href = "./examples/csmsc/tts3">fastspeech2-csmsc</a>
      </td>
    </tr>
   <tr>
      <td rowspan="4">Vocoder</td>
      <td >WaveFlow</td>
      <td >LJSpeech</td>
      <td>
      <a href = "./examples/ljspeech/voc0">waveflow-ljspeech</a>
      </td>
    </tr>
    <tr>
      <td rowspan="3">Parallel WaveGAN</td>
      <td >LJSpeech</td>
      <td>
      <a href = "./examples/ljspeech/voc1">PWGAN-ljspeech</a>
      </td>
    </tr>
    <tr>
      <td >VCTK</td>
      <td>
      <a href = "./examples/vctk/voc1">PWGAN-vctk</a>
      </td>
    </tr>
    <tr>
      <td >CSMSC</td>
      <td>
      <a href = "./examples/csmsc/voc1">PWGAN-csmsc</a>
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


## Tutorials

Normally, [Speech SoTA](https://paperswithcode.com/area/speech) gives you an overview of the hot academic topics in speech. If you want to focus on the two tasks in PaddleSpeech, you will find the following guidelines are helpful to grasp the core ideas.

The original ASR module is based on [Baidu's DeepSpeech](https://arxiv.org/abs/1412.5567) which is an independent product named [DeepSpeech](https://deepspeech.readthedocs.io). However, the toolkit aligns almost all the SoTA modules in the pipeline. Specifically, these modules are

* [Data Prepration](docs/source/asr/data_preparation.md)  
* [Data Augmentation](docs/source/asr/augmentation.md)  
* [Ngram LM](docs/source/asr/ngram_lm.md)  
* [Benchmark](docs/source/asr/benchmark.md)  
* [Relased Model](docs/source/asr/released_model.md)  

The TTS module is originally called [Parakeet](https://github.com/PaddlePaddle/Parakeet), and now merged with DeepSpeech. If you are interested in academic research about this function, please see [TTS research overview](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/docs/source/tts#overview). Also, [this document](https://paddleparakeet.readthedocs.io/en/latest/released_models.html) is a good guideline for the pipeline components.


## FAQ and Contributing

You are warmly welcome to submit questions in [discussions](https://github.com/PaddlePaddle/DeepSpeech/discussions) and bug reports in [issues](https://github.com/PaddlePaddle/DeepSpeech/issues)! Also, we highly appreciate if you would like to contribute to this project!

## License

PaddleSpeech is provided under the [Apache-2.0 License](./LICENSE).

## Acknowledgement

PaddleSpeech depends on a lot of open source repos. See [references](docs/source/asr/reference.md) for more information.
