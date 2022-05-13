## 目录
- [1. 简介](#1)
- [2. 特点](#2)
- [3. 使用教程](#3)
    - [3.1 预训练模型](#31)
    - [3.2 模型训练](#32)
    - [3.3 模型推理](#33)
    - [3.4 服务部署](#33)
    - [3.5 支持个性化场景部署](#33)
- [4. 快速开始](#4)

<a name="1"></a>
## 1. 简介

PP-ASR 是一个 提供 ASR 功能的工具。其提供了多种中文和英文的模型，支持模型的训练，并且支持使用命令行的方式进行模型的推理。 PP-ASR也支持流式模型的部署，以及个性化场景的部署。

<a name="2"></a>
## 2. 特点
语音识别的基本流程如下图所示：  
<center><img src=https://user-images.githubusercontent.com/87408988/168259962-cbe2008b-47b6-443d-9566-d77a5ca2eb25.png width="800" ></center>


PP-ASR 的主要特点如下：
-  提供在中/英文开源数据集 aishell （中文），wenetspeech（中文），librispeech （英文）上的预训练模型。模型包含 deepspeech2 模型以及 conformer/transformer 模型。
-  支持中/英文的模型训练功能。
-  支持命令行方式的模型推理， `paddlespeech asr --input xxx.wav` 方式调用各个预训练模型进行推理。
-  支持流式 ASR 的服务部署，也支持输出时间戳。
-  支持个性化场景的部署。

<a name="3"></a>
## 3. 使用教程

<a name="31"></a>
## 3.1 预训练模型
支持的预训练模型列表：[released_model.md](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md)。
其中效果较好的模型为 Ds2 Online Wenetspeech ASR0 Model 以及 Conformer Online Wenetspeech ASR1 Model。 两个模型都支持流式 ASR。


<a name="32"></a>
## 3.2 模型训练

模型的训练的参考脚本存放在 examples 中，并按照 `examples/数据集/模型` 存放，数据集主要支持 aishell 和 librispeech，模型支持 deepspeech2 模型和 u2 (conformer/transformer) 模型。
具体的执行脚本的步骤记录在 run.sh 当中。具体可参考[这里](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell/asr1)


<a name="33"></a>
## 3.3 模型推理

PPASR 支持在使用`pip install paddlespeech`后 使用命令行的方式来使用预训练模型进行推理。

具体支持的功能包括：

- 对单条音频进行预测
- 使用管道的方式对多条音频进行预测
- 支持 RTF 的计算

具体的使用方式可以参考[这里](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/speech_recognition/README_cn.md) 


<a name="34"></a>
## 3.4 服务部署

PPASR 支持流式ASR的服务部署。支持 语音识别 + 标点处理两个功能同时使用。

server 的 demo [链接](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/streaming_asr_server)

![image](https://user-images.githubusercontent.com/87408988/168255342-1fc790c0-16f4-4540-a861-db239076727c.png)

<a name="35"></a>
## 3.5 支持个性化场景部署

针对个性化场景部署，提供了 特征提取（fbank） => 推理模型（打分库）=> TLG（WFST， token, lexion, grammer）的 C++ 程序。具体参考[这里](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/speechx)

<a name="4"></a>
## 4. 快速开始

关于如果使用 PPASR，可以看这里的[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)，其中提供了 **简单**、**中等**、**困难** 三种安装方式。如果想体验paddlespeech 的推理功能，可以用 **简单** 安装方式。


