# DeepSpeech on PaddlePaddle

[English](README.md)

*DeepSpeech on PaddlePaddle*是一个采用[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)平台的端到端自动语音识别（ASR）引擎的开源项目，
我们的愿景是为语音识别在工业应用和学术研究上，提供易于使用、高效和可扩展的工具，包括训练，推理，测试模块，以及 demo 部署。同时，我们还将发布一些预训练好的英语和普通话模型。


## 模型
* [Baidu's Deep Speech2](http://proceedings.mlr.press/v48/amodei16.pdf)

## 安装
* python3.7
* paddlepaddle 2.0.0

- 安装依赖

```bash
git clone https://github.com/PaddlePaddle/DeepSpeech.git
cd DeepSpeech
pushd tools; make; popd
source tools/venv/bin/activate
bash setup.sh
```

- 开始实验前要source环境.

```bash
source tools/venv/bin/activate
```

## 开始

请查看 [Getting Started](docs/geting_started.md) 和 [tiny egs](examples/tiny/README.md)。

## 问题和帮助

欢迎您在[Github问题](https://github.com/PaddlePaddle/models/issues)中提交问题和bug。也欢迎您为这个项目做出贡献。
