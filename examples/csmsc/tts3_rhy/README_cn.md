(简体中文|[English](./README.md))
# 用 CSMSC 数据集训练 FastSpeech2 模型

本用例包含用于训练 [Fastspeech2](https://arxiv.org/abs/2006.04558) 模型的代码，使用 [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html) 数据集。

## 数据集
### 下载并解压
从 [官方网站](https://test.data-baker.com/data/index/TNtts/) 下载数据集

### 获取MFA结果并解压
我们使用 [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) 去获得 fastspeech2 的音素持续时间。
你们可以从这里直接下载训练好的带节奏时长的 MFA 结果 [baker_alignment_tone.zip](https://paddlespeech.bj.bcebos.com/Rhy_e2e/baker_alignment_tone.zip), 或参考 [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) 训练你自己的模型。
利用 mfa repo 去训练自己的模型时，请添加 `--rhy-with-duration` 。

## 开始
假设数据集的路径是 `~/datasets/BZNSYP`.
假设CSMSC的MFA结果路径为 `./baker_alignment_tone`.
运行下面的命令会进行如下操作：

1. **设置原路径**。
2. 对数据集进行预处理。
3. 训练模型
4. 合成波形
    - 从 `metadata.jsonl` 合成波形。
    - 从文本文件合成波形。
5. 使用静态模型进行推理。
```bash
./run.sh
```
您可以选择要运行的一系列阶段，或者将 `stage` 设置为 `stop-stage` 以仅使用一个阶段，例如，运行以下命令只会预处理数据集。
```bash
./run.sh --stage 0 --stop-stage 0
```
### 数据预处理
```bash
./local/preprocess.sh ${conf_path}
```
当它完成时。将在当前目录中创建 `dump` 文件夹。转储文件夹的结构如下所示。

```text
dump
├── dev
│   ├── norm
│   └── raw
├── phone_id_map.txt
├── speaker_id_map.txt
├── test
│   ├── norm
│   └── raw
└── train
    ├── energy_stats.npy
    ├── norm
    ├── pitch_stats.npy
    ├── raw
    └── speech_stats.npy
```

数据集分为三个部分，即 `train` 、 `dev` 和 `test` ，每个部分都包含一个 `norm` 和 `raw` 子文件夹。原始文件夹包含每个话语的语音、音调和能量特征，而 `norm` 文件夹包含规范化的特征。用于规范化特征的统计数据是从 `dump/train/*_stats.npy` 中的训练集计算出来的。

此外，还有一个 `metadata.jsonl` 在每个子文件夹中。它是一个类似表格的文件，包含音素、文本长度、语音长度、持续时间、语音特征路径、音调特征路径、能量特征路径、说话人和每个话语的 id。

# 更多训练细节请参考 example 下的 CSMSC(tts3)

## 预训练模型
预先训练的端到端带韵律预测的 FastSpeech2 模型：
- [rhy_e2e_pretrain.zip](https://paddlespeech.bj.bcebos.com/Rhy_e2e/rhy_e2e_pretrain.zip)

FastSpeech2检查点包含下列文件。
```text
fastspeech2_nosil_baker_ckpt_0.4
├── default.yaml            # 用于训练 fastspeech2 的默认配置
├── phone_id_map.txt        # 训练 fastspeech2 时的音素词汇文件
├── snapshot_iter_153000.pdz # 模型参数和优化器状态
├── durations.txt           # preprocess.sh的中间过程
├── energy_stats.npy
├── pitch_stats.npy
└── speech_stats.npy        # 训练 fastspeech2 时用于规范化频谱图的统计数据
