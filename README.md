[中文版](README_cn.md)

# PaddlePaddle ASR toolkit

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)

*PaddleASR* is an open-source implementation of end-to-end Automatic Speech Recognition (ASR) engine, with [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) platform. Our vision is to empower both industrial application and academic research on speech recognition, via an easy-to-use, efficient, samller and scalable implementation, including training, inference & testing module, and deployment.


## Models

* [Baidu's DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf)
* [Transformer](https://arxiv.org/abs/1706.03762)
* [Conformer](https://arxiv.org/abs/2005.08100)
* [U2](https://arxiv.org/pdf/2012.05481.pdf)

## Setup

* python>=3.7
* paddlepaddle>=2.1.0

Please see [install](doc/src/install.md).

## Getting Started

Please see [Getting Started](doc/src/getting_started.md) and [tiny egs](examples/tiny/s0/README.md).


## More Information  

* [Install](doc/src/install.md)  
* [Getting Started](doc/src/getting_started.md)  
* [Data Prepration](doc/src/data_preparation.md)  
* [Data Augmentation](doc/src/augmentation.md)  
* [Ngram LM](doc/src/ngram_lm.md)  
* [Server Demo](doc/src/server.md)  
* [Benchmark](doc/src/benchmark.md)  
* [Relased Model](doc/src/released_model.md)  
* [FAQ](doc/src/faq.md)  


## Questions and Help

You are welcome to submit questions in [Github Discussions](https://github.com/PaddlePaddle/DeepSpeech/discussions) and bug reports in [Github Issues](https://github.com/PaddlePaddle/DeepSpeech/issues). You are also welcome to contribute to this project.


## License

DeepSpeech is provided under the [Apache-2.0 License](./LICENSE).

## Acknowledgement

We depends on many open source repos. See [References](doc/src/reference.md) for more information.
