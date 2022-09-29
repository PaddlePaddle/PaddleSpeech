# 背景

模型任务与模型间接请参见 examples/esc50, 本目录是为了校验和测试 paddle.audio 的feature, backend等相关模块而建立.

## 数据集

[TESS: Toronto emotional speech set](https://tspace.library.utoronto.ca/handle/1807/24487) 是一个包含有 200 个目标词的时长为 2 ~ 3 秒的音频,七种情绪的数据集。由两个女演员录制(24岁和64岁),其中情绪分别是愤怒,恶心,害怕,高兴,惊喜,伤心,平淡.

## 模型指标

根据 `TESS` 提供的fold信息，对数据集进行 5-fold 的 fine-tune 2 epoch 训练和评估，dev准确率如下：

|Model|feat_type|Acc|
|--|--|--|
|CNN14| mfcc | 0.8304 |
|CNN14| logmelspectrogram | 0.9893 |
|CNN14| spectrogram| 0.1304 | 
|CNN14| melspectrogram| 0.1339 | 

因为是功能验证,所以只config中训练了 2 个epoch.
log_melspectrogram feature 在迭代 3 个epoch后, acc可以达到0.9983%.
mfcc feature 在迭代3个epoch后, acc可以达到0.9983%.
spectrogram feature 在迭代11个epoch后,acc可达0.95%.
melspectrogram feature 在迭代17个epoch后,acc可到0.9375%.

### 模型训练

启动训练:
```shell
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 1 conf/panns_mfcc.yaml
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 1 conf/panns_logmelspectrogram.yaml
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 1 conf/panns_melspectrogram.yaml
$ CUDA_VISIBLE_DEVICES=0 ./run.sh 1 conf/panns_pectrogram.yaml
```
