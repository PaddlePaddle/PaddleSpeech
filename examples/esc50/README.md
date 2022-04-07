# 声音分类

声音分类和检测是声音算法的一个热门研究方向。  

对于声音分类任务，传统机器学习的一个常用做法是首先人工提取音频的时域和频域的多种特征并做特征选择、组合、变换等，然后基于SVM或决策树进行分类。而端到端的深度学习则通常利用深度网络如RNN，CNN等直接对声间波形(waveform)或时频特征(time-frequency)进行特征学习(representation learning)和分类预测。

在IEEE ICASSP 2017 大会上，谷歌开放了一个大规模的音频数据集[Audioset](https://research.google.com/audioset/)。该数据集包含了 632 类的音频类别以及 2,084,320 条人工标记的每段 **10 秒**长度的声音剪辑片段（来源于YouTube视频）。目前该数据集已经有 210万 个已标注的视频数据，5800 小时的音频数据，经过标记的声音样本的标签类别为 527。

`PANNs`([PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/pdf/1912.10211.pdf))是基于Audioset数据集训练的声音分类/识别的模型。经过预训练后，模型可以用于提取音频的embbedding。本示例将使用`PANNs`的预训练模型Finetune完成声音分类的任务。


## 模型简介

PaddleAudio提供了PANNs的CNN14、CNN10和CNN6的预训练模型，可供用户选择使用：
- CNN14: 该模型主要包含12个卷积层和2个全连接层，模型参数的数量为 79.6M，embbedding维度是 2048。
- CNN10: 该模型主要包含8个卷积层和2个全连接层，模型参数的数量为 4.9M，embbedding维度是 512。
- CNN6: 该模型主要包含4个卷积层和2个全连接层，模型参数的数量为 4.5M，embbedding维度是 512。


## 数据集

[ESC-50: Dataset for Environmental Sound Classification](https://github.com/karolpiczak/ESC-50) 是一个包含有 2000 个带标签的时长为 **5 秒**的环境声音样本，音频样本采样率为 44,100Hz 的单通道音频文件，所有样本根据标签被划分为 50 个类别，每个类别有 40 个样本。

## 模型指标

根据 `ESC-50` 提供的fold信息，对数据集进行 5-fold 的 fine-tune 训练和评估，平均准确率如下：

|Model|Acc|
|--|--|
|CNN14| 0.9500
|CNN10| 0.8975
|CNN6| 0.8825

## 快速开始

### 模型训练

运行下面的命令，可在训练集上进行模型的finetune，支持单机的单卡训练和多卡训练。

启动训练:
```shell
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 1 conf/panns.yaml
```

训练的参数可在 `conf/panns.yaml` 的 `training` 中配置，其中：
- `epochs`: 训练轮次，默认为 50。
- `learning_rate`: Fine-tune的学习率；默认为5e-5。
- `batch_size`: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 16。
- `num_workers`: Dataloader获取数据的子进程数。默认为0，加载数据的流程在主进程执行。
- `checkpoint_dir`: 模型参数文件和optimizer参数文件的保存目录，默认为`./checkpoint`。
- `save_freq`: 训练过程中的模型保存频率，默认为 10。
- `log_freq`: 训练过程中的信息打印频率，默认为 10。

示例代码中使用的预训练模型为`CNN14`，如果想更换为其他预训练模型，可通过修改 `conf/panns.yaml` 的 `model` 中配置：
```yaml
# CNN14
model:
  backbone: 'paddlespeech.cls.models:cnn14'
```
```yaml
# CNN10
model:
  backbone: 'paddlespeech.cls.models:cnn10'
```
```yaml
# CNN6
model:
  backbone: 'paddlespeech.cls.models:cnn6'
```

### 模型预测

```shell
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 2 conf/panns.yaml
```

训练的参数可在 `conf/panns.yaml` 的 `predicting` 中配置，其中：
- `audio_file`: 指定预测的音频文件。
- `top_k`: 预测显示的top k标签的得分，默认为 1。
- `checkpoint`: 模型参数checkpoint文件。

输出的预测结果如下：
```
[/audio/dog.wav]
Dog: 0.9999538660049438
Clock tick: 1.3341237718123011e-05
Cat: 6.579841738130199e-06
```

### 模型部署

#### 1. 动转静

模型训练结束后，可以将已保存的动态图参数导出成静态图的模型和参数，然后实施静态图的部署。

```shell
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 3 ./checkpoint/epoch_50/model.pdparams ./export
```

`paddlespeech/cls/exps/panns/export_model.py` 脚本中可支持配置的参数：
- `checkpoint`: 模型参数checkpoint文件。
- `output_dir`: 导出静态图模型和参数文件的保存目录。

导出的静态图模型和参数文件如下：
```sh
$ tree export
export
├── inference.pdiparams
├── inference.pdiparams.info
└── inference.pdmodel
```

#### 2. 模型部署和预测

`paddlespeech/cls/exps/panns/deploy/predict.py` 脚本使用了`paddle.inference`模块下的api，提供了python端部署的示例：

```shell
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 4 cpu ./export /audio/dog.wav
```

`paddlespeech/cls/exps/panns/deploy/predict.py` 脚本中可支持配置的主要参数：
- `device`: 指定模型预测时使用的设备。
- `model_dir`: 导出静态图模型和参数文件的保存目录。
- `wav`: 指定预测的音频文件。

## Reference
* [PANNs(PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition)](https://arxiv.org/abs/1912.10211)
