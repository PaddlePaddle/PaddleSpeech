English | [简体中文](README_ch.md)

# PaddleSpeech



<p align="center">
  <img src="./docs/images/PaddleSpeech_log.png" />
</p>
<div align="center">  

  <h3> 
  <a href="https://github.com/Mingxue-Xu/DeepSpeech#quick-start"> Quick Start </a> 
  | <a href="https://github.com/Mingxue-Xu/DeepSpeech#tutorials"> Tutorials </a> 
  | <a href="https://github.com/Mingxue-Xu/DeepSpeech#model-list"> Models List </a> 
  
</div>
  
------------------------------------------------------------------------------------
![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)

> Notes: Is there any idea to add [Parakeet logo](https://github.com/PaddlePaddle/Parakeet/blob/develop/docs/images/logo.png) into this .md document?

<!---
why they should use your module, 
how they can install it, 
how they can use it
-->

**PaddleSpeech** is an open-source toolkit on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) platform for two critical tasks in Speech - Automatic Speech Recognition (ASR) and Text-To-Speech Synthesis (TTS), with modules involving state-of-art and influential models. 

Via the easy-to-use, efficient, flexible and scalable implementation, our vision is to empower both industrial application and academic research, including training, inference & testing module, and deployment. Besides, this toolkit also features at:
- **Rule-based Chinese frontend**: we utilize plenty of Chinese datasets and corpora to enhance user experience, including CSMSC and Baidu Internal Corpus.
- **Supporting of ASR streaming and non-streaming data**: This toolkit contains non-streaming models like [Baidu's DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf), [Transformer](https://arxiv.org/abs/1706.03762) and [Conformer](https://arxiv.org/abs/2005.08100). And for streaming models, we have [Baidu's DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf) and [U2](https://arxiv.org/pdf/2012.05481.pdf).
- **Varieties of mainstream models**: The toolkit integrates modules that participate in the whole pipeline of both ASR and TTS, [See also model lists](#models-list).
  
> Notes: It is better to add a brief getting started.

## Table of Contents

The contents of this README is as follow:

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models List](#models-list)
- [Tutorials](#tutorials)
- [FAQ and Contributing](#faq-and-contributing)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Installation

> Note: The installation guidance of TTS and ASR is now separated.

Base environment:  
* Ubuntu 16.04
* python>=3.7
* paddlepaddle==2.1.2

Please see the [ASR installation](docs/source/asr/install.md) and [TTS installation](docs/source/tts/install.md) documents for all the alternatives.

## Quick Start

> Note: It is better to use code blocks rather than hyperlinks.

Please see [ASR getting started](docs/source/asr/getting_started.md) ([tiny test](examples/tiny/s0/README.md)) and [TTS Basic Use](/docs/source/tts/basic_usage.md).

## Models List

PaddleSpeech ASR supports a lot of mainstream models, which are summarized as follow. For more information, please refer to [ASRModels](./docs/source/asr/released_model.md).

<!---
The current hyperlinks redirect to [Previous Parakeet](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples). 
-->

<table>
  <thead>
    <tr>
      <th>ASR Module Type</th>
      <th>Model Type</th>
      <th>Dataset</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Acoustic Model</td>
      <td >2 Conv + 5 LSTM layers with only forward direction	</td>
      <td rowspan="4" >Aishell</td>
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
      <td >Encoder:Conformer, Decoder:Transformer, Decoding method: Attention</td>
      <td rowspan="2"> Librispeech</td>
      <td> <a href = "https://deepspeech.bj.bcebos.com/release2.1/librispeech/s1/conformer.release.tar.gz">Conformer Librispeech Model</a> </td>
    </tr>
    <tr>
      <td>Encoder:Conformer, Decoder:Transformer, Decoding method: Attention</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/release2.1/librispeech/s1/transformer.release.tar.gz">Transformer Librispeech Model</a>
      </td>
    </tr>
   <tr>
      <td rowspan="3">Language Model</td>
      <td >English LM</td>
      <td >CommonCrawl(en.00)</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm">English LM</a>
      </td>
    </tr>
    <tr>
      <td>Mandarin LM Small</td>
      <td rowspan="2">Baidu Internal Corpus</td>
      <td>
      <a href = "https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm">Mandarin LM Small</a>
      </td>
    </tr>
    <tr>
      <td >Mandarin LM Large</td>
      <td >
      <a href = "https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm">Mandarin LM Large</a>
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
      <td rowspan="6">Acoustic Model</td>
      <td >Tacotron2</td>
      <td rowspan="2" >LJSpeech</td>
      <td>
      <a href = "https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/fastspeech2/vctk">tacotron2-vctk</a>
      </td>
    </tr>
    <tr>
      <td>TransformerTTS</td>
      <td>
      <a href = "https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/transformer_tts/ljspeech">transformer-ljspeech</a>
      </td>
    </tr>
    <tr>
      <td>SpeedySpeech</td>
      <td>CSMSC</td>
      <td >
      <a href = "https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/speedyspeech/baker">speedyspeech-csmsc</a>
      </td>
    </tr>
    <tr>
      <td rowspan="3">FastSpeech2</td>
      <td>AISHELL-3</td>
      <td>
      <a href = "https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/fastspeech2/aishell3">fastspeech2-aishell3</a>
      </td>
    </tr>
      <tr>
      <td>VCTK</td>
      <td> <a href = "https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/fastspeech2/vctk">fastspeech2-vctk</a> </td>
    </tr>
    <tr>
      <td>CSMSC</td>
      <td>
      <a href = "https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/fastspeech2/baker">fastspeech2-csmsc</a>
      </td>
    </tr>
   <tr>
      <td rowspan="3">Vocoder</td>
      <td >WaveFlow</td>
      <td >LJSpeech</td>
      <td>
      <a href = "https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/GANVocoder/parallelwave_gan/ljspeech">waveflow-ljspeech</a>
      </td>
    </tr>
    <tr>
      <td rowspan="2">Parallel WaveGAN</td>
      <td >LJSpeech</td>
      <td>
      <a href = "https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/fastspeech2/baker">pwGAN-ljspeech</a>
      </td>
    </tr>
    <tr>
      <td >CSMSC</td>
      <td>
      <a href = "https://github.com/PaddlePaddle/Parakeet/tree/develop/examples/GANVocoder/parallelwave_gan/baker">pwGAN-csmsc</a>
      </td>
    </tr>
  </tbody>
</table>


## Tutorials 

More background information for ASR, please refer to:

* [Data Prepration](docs/source/asr/data_preparation.md)  
* [Data Augmentation](docs/source/asr/augmentation.md)  
* [Ngram LM](docs/source/asr/ngram_lm.md)  
* [Benchmark](docs/source/asr/benchmark.md)  
* [Relased Model](docs/source/asr/released_model.md)  

For TTS, [this document](https://paddleparakeet.readthedocs.io/en/latest/) is a good guideline.


## FAQ and Contributing

You are warmly welcome to submit questions in [Discussions](https://github.com/PaddlePaddle/DeepSpeech/discussions) and bug reports in [Issues](https://github.com/PaddlePaddle/DeepSpeech/issues)!

Also, we highly appreciate if you would like to contribute to this project!

## License

DeepSpeech is provided under the [Apache-2.0 License](./LICENSE).

## Acknowledgement

DeepSpeech depends on many open source repos. See [References](docs/source/asr/reference.md) for more information.

<code> **Updates on 2021/10/21**: This [README.md](README.md) outline is not completed, especially *from section **Quick Start***.</code>


