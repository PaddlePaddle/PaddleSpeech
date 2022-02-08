(简体中文|[English](./README.md))
# 用 CSMSC 数据集训练 FastSpeech2 模型

本用例包含用于训练 [Fastspeech2](https://arxiv.org/abs/2006.04558) 模型的代码，使用 [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html) 数据集。

## 数据集
### 下载并解压
从 [官方网站](https://test.data-baker.com/data/index/source) 下载数据集

### 获取MFA结果并解压
我们使用 [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) 去获得 fastspeech2 的音素持续时间。
你们可以从这里下载 [baker_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz), 或参考 [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) 训练你自己的模型。

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
                [--speaker-dict SPEAKER_DICT] [--voice-cloning VOICE_CLONING]

Train a FastSpeech2 model.

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
  --voice-cloning VOICE_CLONING
                        whether training voice cloning model.
```
1. `--config` 是一个 yaml 格式的配置文件，用于覆盖默认配置，位于 `conf/default.yaml`.
2. `--train-metadata` 和 `--dev-metadata` 应为 `dump` 文件夹中 `train` 和 `dev` 下的规范化元数据文件
3. `--output-dir` 是保存结果的目录。 检查点保存在此目录中的 `checkpoints/` 目录下。
4. `--ngpu` 要使用的 GPU 数，如果 ngpu==0，则使用 cpu 。
5. `--phones-dict` 是音素词汇表文件的路径。

### 合成
我们使用 [parallel wavegan](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc1) 作为神经声码器（vocoder）。
从 [pwg_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip) 下载预训练的 parallel wavegan 模型并将其解压。

```bash
unzip pwg_baker_ckpt_0.4.zip
```
Parallel WaveGAN 检查点包含如下文件。
```text
pwg_baker_ckpt_0.4
├── pwg_default.yaml               # 用于训练 parallel wavegan 的默认配置
├── pwg_snapshot_iter_400000.pdz   # parallel wavegan 的模型参数
└── pwg_stats.npy                  # 训练平行波形时用于规范化谱图的统计数据
```
`./local/synthesize.sh` 调用 `${BIN_DIR}/../synthesize.py` 即可从 `metadata.jsonl`中合成波形。

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h]
                     [--am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk}]
                     [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                     [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                     [--tones_dict TONES_DICT] [--speaker_dict SPEAKER_DICT]
                     [--voice-cloning VOICE_CLONING]
                     [--voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc}]
                     [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                     [--voc_stat VOC_STAT] [--ngpu NGPU]
                     [--test_metadata TEST_METADATA] [--output_dir OUTPUT_DIR]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk}
                        Choose acoustic model type of tts task.
  --am_config AM_CONFIG
                        Config of acoustic model. Use deault config when it is
                        None.
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
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc}
                        Choose vocoder type of tts task.
  --voc_config VOC_CONFIG
                        Config of voc. Use deault config when it is None.
  --voc_ckpt VOC_CKPT   Checkpoint file of voc.
  --voc_stat VOC_STAT   mean and standard deviation used to normalize
                        spectrogram when training voc.
  --ngpu NGPU           if ngpu == 0, use cpu.
  --test_metadata TEST_METADATA
                        test metadata.
  --output_dir OUTPUT_DIR
                        output dir.
```
`./local/synthesize_e2e.sh` 调用 `${BIN_DIR}/../synthesize_e2e.py`，即可从文本文件中合成波形。

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize_e2e.py [-h]
                         [--am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk}]
                         [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                         [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                         [--tones_dict TONES_DICT]
                         [--speaker_dict SPEAKER_DICT] [--spk_id SPK_ID]
                         [--voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc}]
                         [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                         [--voc_stat VOC_STAT] [--lang LANG]
                         [--inference_dir INFERENCE_DIR] [--ngpu NGPU]
                         [--text TEXT] [--output_dir OUTPUT_DIR]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk}
                        Choose acoustic model type of tts task.
  --am_config AM_CONFIG
                        Config of acoustic model. Use deault config when it is
                        None.
  --am_ckpt AM_CKPT     Checkpoint file of acoustic model.
  --am_stat AM_STAT     mean and standard deviation used to normalize
                        spectrogram when training acoustic model.
  --phones_dict PHONES_DICT
                        phone vocabulary file.
  --tones_dict TONES_DICT
                        tone vocabulary file.
  --speaker_dict SPEAKER_DICT
                        speaker id map file.
  --spk_id SPK_ID       spk id for multi speaker acoustic model
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc}
                        Choose vocoder type of tts task.
  --voc_config VOC_CONFIG
                        Config of voc. Use deault config when it is None.
  --voc_ckpt VOC_CKPT   Checkpoint file of voc.
  --voc_stat VOC_STAT   mean and standard deviation used to normalize
                        spectrogram when training voc.
  --lang LANG           Choose model language. zh or en
  --inference_dir INFERENCE_DIR
                        dir to save inference models
  --ngpu NGPU           if ngpu == 0, use cpu.
  --text TEXT           text to synthesize, a 'utt_id sentence' pair per line.
  --output_dir OUTPUT_DIR
                        output dir.
```
1. `--am` 声学模型格式是否符合 {model_name}_{dataset}
2. `--am_config`, `--am_checkpoint`, `--am_stat` 和 `--phones_dict` 是声学模型的参数，对应于 fastspeech2 预训练模型中的 4 个文件。
3. `--voc` 声码器(vocoder)格式是否符合 {model_name}_{dataset}
4. `--voc_config`, `--voc_checkpoint`, `--voc_stat` 是声码器的参数，对应于 parallel wavegan 预训练模型中的 3 个文件。
5. `--lang` 对应模型的语言可以是 `zh` 或 `en` 。
6. `--test_metadata` 应为 `dump` 文件夹中 `test` 下的规范化元数据文件、
7. `--text` 是文本文件，其中包含要合成的句子。
8. `--output_dir` 是保存合成音频文件的目录。
9. `--ngpu` 要使用的GPU数，如果 ngpu==0，则使用 cpu 。

### 推理
在合成之后，我们将在 `${train_output_path}/inference` 中得到 fastspeech2 和 pwgan 的静态模型
`./local/inference.sh` 调用 `${BIN_DIR}/inference.py` 为 fastspeech2 + pwgan 综合提供了一个 paddle 静态模型推理示例。

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${train_output_path}
```

## 预训练模型
预先训练的 FastSpeech2 模型，在音频边缘没有空白音频：
- [fastspeech2_nosil_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip)
- [fastspeech2_conformer_baker_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_conformer_baker_ckpt_0.5.zip)

静态模型可以在这里下载 [fastspeech2_nosil_baker_static_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_static_0.4.zip).

Model | Step | eval/loss | eval/l1_loss | eval/duration_loss | eval/pitch_loss| eval/energy_loss 
:-------------:| :------------:| :-----: | :-----: | :--------: |:--------:|:---------:
default| 2(gpu) x 76000|1.0991|0.59132|0.035815|0.31915|0.15287|
conformer| 2(gpu) x 76000|1.0675|0.56103|0.035869|0.31553|0.15509|

FastSpeech2检查点包含下列文件。
```text
fastspeech2_nosil_baker_ckpt_0.4
├── default.yaml            # 用于训练 fastspeech2 的默认配置
├── phone_id_map.txt        # 训练 fastspeech2 时的音素词汇文件
├── snapshot_iter_76000.pdz # 模型参数和优化器状态
└── speech_stats.npy        # 训练 fastspeech2 时用于规范化频谱图的统计数据
```
您可以使用以下脚本通过使用预训练的 fastspeech2 和 parallel wavegan 模型为 `${BIN_DIR}/../sentences.txt` 合成句子
```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize_e2e.py \
  --am=fastspeech2_csmsc \
  --am_config=fastspeech2_nosil_baker_ckpt_0.4/default.yaml \
  --am_ckpt=fastspeech2_nosil_baker_ckpt_0.4/snapshot_iter_76000.pdz \
  --am_stat=fastspeech2_nosil_baker_ckpt_0.4/speech_stats.npy  \
  --voc=pwgan_csmsc \
  --voc_config=pwg_baker_ckpt_0.4/pwg_default.yaml \
  --voc_ckpt=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
  --voc_stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
  --lang=zh \
  --text=${BIN_DIR}/../sentences.txt \
  --output_dir=exp/default/test_e2e \
  --inference_dir=exp/default/inference \
  --phones_dict=fastspeech2_nosil_baker_ckpt_0.4/phone_id_map.txt
```
