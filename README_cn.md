[English](README.md)

# PaddlePaddle ASR toolkit

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)

*PaddleASR*是一个采用[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)平台的端到端自动语音识别（ASR）引擎的开源项目，
我们的愿景是为语音识别在工业应用和学术研究上，提供易于使用、高效、小型化和可扩展的工具，包括训练，推理，以及  部署。

## 模型

* [Baidu's DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf)
* [Transformer](https://arxiv.org/abs/1706.03762)
* [Conformer](https://arxiv.org/abs/2005.08100)
* [U2](https://arxiv.org/pdf/2012.05481.pdf)


## 安装

* python>=3.7
* paddlepaddle>=2.1.0

参看 [安装](doc/install.md)。

## 开始

请查看 [Getting Started](doc/src/getting_started.md) 和 [tiny egs](examples/tiny/README.md)。

## 更多信息

* [安装](doc/src/install.md)  
* [开始](doc/src/getting_started.md)  
* [数据处理](doc/src/data_preparation.md)  
* [数据增强](doc/src/augmentation.md)  
* [语言模型](doc/src/ngram_lm.md)  
* [服务部署](doc/src/server.md)  
* [Benchmark](doc/src/benchmark.md)  
* [Relased Model](doc/src/released_model.md)  
* [FAQ](doc/src/faq.md)  

## 问题和帮助

欢迎您在[Github讨论](https://github.com/PaddlePaddle/DeepSpeech/discussions)提交问题，[Github问题](https://github.com/PaddlePaddle/models/issues)中反馈bug。也欢迎您为这个项目做出贡献。

## License

DeepSpeech遵循[Apache-2.0开源协议](./LICENSE)。

## 感谢

开发中参考一些优秀的仓库，详情参见 [References](doc/src/reference.md)。
