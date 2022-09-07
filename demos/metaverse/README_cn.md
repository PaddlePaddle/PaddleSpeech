(简体中文|[English](./README.md))

# Metaverse

## 简介

Metaverse是一种新的互联网应用和社交形式，融合了多种新技术，产生了虚拟现实。

这个演示是一个让图片中的名人“说话”的实现。通过 `PaddleSpeech` 和 `PaddleGAN`的 `TTS` 模块的组合，我们集成了安装和特定模块到一个shell脚本中。

## 使用

您可以使用 `PaddleSpeech` 和`PaddleGAN`的 `TTS` 模块让您最喜欢的人说出指定的内容，并构建您的虚拟人。

运行 `run.sh` 完成所有基本程序，包括安装。

```bash
./run.sh
```

在 `run.sh`, 先会执行 `source path.sh` 来设置好环境变量。

如果您想尝试您的句子，请替换`sentences.txt`中的句子。

如果您想尝试图像，请将图像替换shell脚本中的`download/Lamarr.png`。

结果已显示在我们的 [notebook](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/tutorial/tts/tts_tutorial.ipynb)。
