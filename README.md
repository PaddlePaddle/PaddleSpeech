# PaddleSpeech

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)

<!---
Here place an icon/image as the logo at the beginning like PaddleOCR/PaddleNLP.
Is there any idea to add Parakeet logo(https://github.com/PaddlePaddle/Parakeet/blob/develop/docs/images/logo.png) into this .md document? 
-->

<!---
README.me should include:
why they should use your module, 
how they can install it, 
how they can use it
-->

**PaddleSpeech** is an open-source toolkit on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) platform for two critical tasks in Speech - Automatic Speech Recognition (ASR) and Text-To-Speech Synthesis (TTS), with modules involving state-of-art and influential models.

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Guidelines of DeepSpeech Pipeline](#guidelines-of-deepspeech-pipeline)
- [FAQ and Contributing](#faq-and-contributing)
- [Acknowledgement](#acknowledgement)
- [License](#license) 

## Features

Via the easy-to-use, efficient, flexible and scalable implementation, our vision is to empower both industrial application and academic research, including training, inference & testing module, and deployment.

<!---
1.The following features are summarized from docs/source/asr/feature_list.md, please add the features of Parakeet @yt605155624 :-) 
2.Better add hyperlinks for code path/dir
-->

The features of **ASR** are summarized as follows:
- **Used datasets**
  - Aishell, THCHS30, TIMIT and Librispeech
- **Model support of streaming and non-streaming data**
  - Non-streaming: [Baidu's DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf), [Transformer](https://arxiv.org/abs/1706.03762) and [Conformer](https://arxiv.org/abs/2005.08100)
  - Streaming:  [Baidu's DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf) and [U2](https://arxiv.org/pdf/2012.05481.pdf)
- **Language Model**: Ngram
- **Decoder**: ctc greedy, ctc prefix beam search, greedy, beam search, attention rescore
- **Aligment**: MFA, CTC Aligment
- **Speech Frontend**
  - Audio: Auto Gain
  - Feature: kaldi fbank, kaldi mfcc, linear, delta detla
- **Speech Augmentation**
  - Audio: Auto Gain
  - Feature: Volume Perturbation, Speed Perturbation, Shifting Perturbation, Online Bayesian normalization, Noise Perturbation, Impulse Response,Spectrum, SpecAugment, Adaptive SpecAugment
- **Tokenizer**: Chinese/English Character, English Word, Sentence Piece

- **Word Segmentation**: [mmseg](http://technology.chtsai.org/mmseg/)

The features of **TTS** are summarized as follows:

- **Blabla**
  - Blabla ...

## Installation

All tested under:  
* Ubuntu 16.04
* python>=3.7
* paddlepaddle==2.1.2

Please see the [installation](docs/source/asr/install.md) doc for all the alternatives.

## Getting Started

Please see [Getting Started](docs/source/asr/getting_started.md) and [tiny egs](examples/tiny/s0/README.md).


## Guidelines of Pipeline  

* [Data Prepration](docs/source/asr/data_preparation.md)  
* [Data Augmentation](docs/source/asr/augmentation.md)  
* [Ngram LM](docs/source/asr/ngram_lm.md)  
* [Benchmark](docs/source/asr/benchmark.md)  
* [Relased Model](docs/source/asr/released_model.md)  


## FAQ and Contributing

You are warmly welcome to submit questions in [Discussions](https://github.com/PaddlePaddle/DeepSpeech/discussions) and bug reports in [Issues](https://github.com/PaddlePaddle/DeepSpeech/issues)!

Also, we highly appreciate if you would like to contribute to this project!

## License

DeepSpeech is provided under the [Apache-2.0 License](./LICENSE).

## Acknowledgement

DeepSpeech depends on many open source repos. See [References](docs/source/asr/reference.md) for more information.

<code> **Updates on 2021/10/20**: This [README.md](README.md) outline is not completed, especially for TTS module *from section **Features***. </code>
