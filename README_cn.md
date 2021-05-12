[English](README.md)

# DeepSpeech on PaddlePaddle

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)

*DeepSpeech on PaddlePaddle*是一个采用[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)平台的端到端自动语音识别（ASR）引擎的开源项目，
我们的愿景是为语音识别在工业应用和学术研究上，提供易于使用、高效和可扩展的工具，包括训练，推理，测试模块，以及 demo 部署。同时，我们还将发布一些预训练好的英语和普通话模型。

## 模型

* [Baidu's DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf)
* [Transformer](https://arxiv.org/abs/1706.03762)
* [Conformer](https://arxiv.org/abs/2005.08100)
* [U2](https://arxiv.org/pdf/2012.05481.pdf)


## 安装

* python>=3.7
* paddlepaddle>=2.0.0

参看 [安装](docs/install.md)。

## 开始

请查看 [Getting Started](docs/src/geting_started.md) 和 [tiny egs](examples/tiny/README.md)。

## 更多信息

* [安装](docs/src/install.md)  
* [开始](docs/src/geting_stared.md)  
* [数据处理](docs/src/data_preparation.md)  
* [数据增强](docs/src/augmentation.md)  
* [语言模型](docs/src/ngram_lm.md)  
* [服务部署](docs/src/server.md)  
* [Benchmark](docs/src/benchmark.md)  
* [Relased Model](docs/src/released_model.md)  
* [FAQ](docs/src/faq.md)  

## 问题和帮助

欢迎您在[Github讨论](https://github.com/PaddlePaddle/DeepSpeech/discussions)提交问题，[Github问题](https://github.com/PaddlePaddle/models/issues)中反馈bug。也欢迎您为这个项目做出贡献。

## License

DeepSpeech遵循[Apache-2.0开源协议](./LICENSE)。

## 感谢

开发中参考一些优秀的仓库，详情参见 [References](docs/src/reference.md)。
