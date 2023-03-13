(简体中文|[English](./README.md))
# 用 Opencpop 数据集训练 DiffSinger 模型

本用例包含用于训练 [DiffSinger](https://arxiv.org/abs/2105.02446) 模型的代码，使用 [Mandarin singing corpus](https://wenet.org.cn/opencpop/) 数据集。

## 数据集
### 下载并解压
从 [官方网站](https://wenet.org.cn/opencpop/download/) 下载数据集

## 开始
假设数据集的路径是 `~/datasets/Opencpop`.
运行下面的命令会进行如下操作：

1. **设置原路径**。
2. 对数据集进行预处理。
3. 训练模型
4. 合成波形
    - 从 `metadata.jsonl` 合成波形。
    - （支持中）从文本文件合成波形。
5. （支持中）使用静态模型进行推理。
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
    ├── speech_stats.npy
    └── speech_stretchs.npy
```

数据集分为三个部分，即 `train` 、 `dev` 和 `test` ，每个部分都包含一个 `norm` 和 `raw` 子文件夹。原始文件夹包含每个话语的语音、音调和能量特征，而 `norm` 文件夹包含规范化的特征。用于规范化特征的统计数据是从 `dump/train/*_stats.npy` 中的训练集计算出来的。`speech_stretchs.npy` 中包含 mel谱每个维度上的最小值和最大值，用于 diffusion 模块训练/推理前的线性拉伸。
注意：由于非 norm 特征训练效果由于 norm，因此 `norm` 下保存的特征是未经过 norm 的特征。


此外，还有一个 `metadata.jsonl` 在每个子文件夹中。它是一个类似表格的文件，包含话语id，音色id，音素、文本长度、语音长度、音素持续时间、语音特征路径、音调特征路径、能量特征路径、音调，音调持续时间，是否为转音。

### 模型训练
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
`./local/train.sh` 调用 `${BIN_DIR}/train.py` 。
以下是完整的帮助信息。

```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                [--ngpu NGPU] [--phones-dict PHONES_DICT]
                [--speaker-dict SPEAKER_DICT] [--speech-stretchs SPEECH_STRETCHS]

Train a DiffSinger model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       fastspeech2 config file.
  --train-metadata TRAIN_METADATA
                        training data.
  --dev-metadata DEV_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        output dir.
  --ngpu NGPU           if ngpu=0, use cpu.
  --phones-dict PHONES_DICT
                        phone vocabulary file.
  --speaker-dict SPEAKER_DICT
                        speaker id map file for multiple speaker model.
  --speech-stretchs SPEECH_STRETCHS
                        min amd max mel for stretching.
```
1. `--config` 是一个 yaml 格式的配置文件，用于覆盖默认配置，位于 `conf/default.yaml`.
2. `--train-metadata` 和 `--dev-metadata` 应为 `dump` 文件夹中 `train` 和 `dev` 下的规范化元数据文件
3. `--output-dir` 是保存结果的目录。 检查点保存在此目录中的 `checkpoints/` 目录下。
4. `--ngpu` 要使用的 GPU 数，如果 ngpu==0，则使用 cpu 。
5. `--phones-dict` 是音素词汇表文件的路径。
6. `--speech-stretchs` mel的最小最大值数据的文件路径。

### 合成
我们使用 parallel opencpop 作为神经声码器（vocoder）。
从 [pwgan_opencpop_ckpt_1.4.0.zip](https://paddlespeech.bj.bcebos.com/t2s/svs/opencpop/pwgan_opencpop_ckpt_1.4.0.zip) 下载预训练的 parallel wavegan 模型并将其解压。

```bash
unzip pwgan_opencpop_ckpt_1.4.0.zip
```
Parallel WaveGAN 检查点包含如下文件。
```text
pwgan_opencpop_ckpt_1.4.0.zip
├── default.yaml               # 用于训练 parallel wavegan 的默认配置
├── snapshot_iter_100000.pdz   # parallel wavegan 的模型参数
└── feats_stats.npy            # 训练平行波形时用于规范化谱图的统计数据
```
`./local/synthesize.sh` 调用 `${BIN_DIR}/../synthesize.py` 即可从 `metadata.jsonl`中合成波形。

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h]
                     [--am {diffsinger_opencpop}]
                     [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                     [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                     [--voc {pwgan_opencpop}]
                     [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                     [--voc_stat VOC_STAT] [--ngpu NGPU]
                     [--test_metadata TEST_METADATA] [--output_dir OUTPUT_DIR]
                     [--speech_stretchs SPEECH_STRETCHS]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech,tacotron2_aishell3}
                        Choose acoustic model type of tts task.
  --am_config AM_CONFIG
                        Config of acoustic model.
  --am_ckpt AM_CKPT     Checkpoint file of acoustic model.
  --am_stat AM_STAT     mean and standard deviation used to normalize
                        spectrogram when training acoustic model.
  --phones_dict PHONES_DICT
                        phone vocabulary file.
  --tones_dict TONES_DICT
                        tone vocabulary file.
  --speaker_dict SPEAKER_DICT
                        speaker id map file.
  --voice-cloning VOICE_CLONING
                        whether training voice cloning model.
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,wavernn_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,style_melgan_csmsc}
                        Choose vocoder type of tts task.
  --voc_config VOC_CONFIG
                        Config of voc.
  --voc_ckpt VOC_CKPT   Checkpoint file of voc.
  --voc_stat VOC_STAT   mean and standard deviation used to normalize
                        spectrogram when training voc.
  --ngpu NGPU           if ngpu == 0, use cpu.
  --test_metadata TEST_METADATA
                        test metadata.
  --output_dir OUTPUT_DIR
                        output dir.
  --speech-stretchs     mel min and max values file.
```

## 预训练模型
预先训练的 DiffSinger 模型：
- [diffsinger_opencpop_ckpt_1.4.0.zip](https://paddlespeech.bj.bcebos.com/t2s/svs/opencpop/diffsinger_opencpop_ckpt_1.4.0.zip)


DiffSinger 检查点包含下列文件。
```text
diffsinger_opencpop_ckpt_1.4.0.zip
├── default.yaml             # 用于训练 diffsinger 的默认配置
├── energy_stats.npy         # 训练 diffsinger 时如若需要 norm energy 会使用到的统计数据 
├── phone_id_map.txt         # 训练 diffsinger 时的音素词汇文件
├── pitch_stats.npy          # 训练 diffsinger 时如若需要 norm pitch 会使用到的统计数据 
├── snapshot_iter_160000.pdz # 模型参数和优化器状态
├── speech_stats.npy         # 训练 diffsinger 时用于规范化频谱图的统计数据
└── speech_stretchs.npy      # 训练 diffusion 前用于 mel 谱拉伸的最小及最大值

```
目前文本前端未完善，暂不支持 `synthesize_e2e` 的方式合成音频。尝试效果可先使用 `synthesize`。
