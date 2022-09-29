# 背景

TESS音频情绪分类任务.
从而校验和测试 paddle.audio 的feature, backend等相关模块.

本实验采用了PaddleSpeech提供了PANNs的CNN14的预训练模型进行finetune：
- CNN14: 该模型主要包含12个卷积层和2个全连接层，模型参数的数量为 79.6M，embbedding维度是 2048。

`PANNs`([PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/pdf/1912.10211.pdf))是基于Audioset数据集训练的声音分类/识别的模型。经过预训练后，模型可以用于提取音频的embbedding。本示例将使用`PANNs`的预训练模型Finetune完成声音分类的任务。

## 数据集

[TESS: Toronto emotional speech set](https://tspace.library.utoronto.ca/handle/1807/24487) 是一个包含有 200 个目标词的时长为 2 ~ 3 秒的音频,七种情绪的数据集。由两个女演员录制(24岁和64岁),其中情绪分别是愤怒,恶心,害怕,高兴,惊喜,伤心,平淡.

## 模型指标

根据 `TESS` 提供的fold信息，对数据集进行 5-fold 的 fine-tune 训练和评估，dev准确率如下：

|Model|feat_type|Acc| note |
|--|--|--| -- |
|CNN14| mfcc | 0.9929 |3 epoch |
|CNN14| logmelspectrogram | 0.9983 | 3 epoch |
|CNN14| spectrogram| 0.95 | 11 epoch |
|CNN14| melspectrogram| 0.9375 | 17 epoch |

### 模型训练

启动训练:
```shell
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 1 conf/panns_mfcc.yaml
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 1 conf/panns_logmelspectrogram.yaml
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 1 conf/panns_melspectrogram.yaml
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 1 conf/panns_pectrogram.yaml
```
