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
  --config CONFIG       diffsinger config file.
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
       {diffsinger_opencpop} Choose acoustic model type of svs task.
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
        {pwgan_opencpop, hifigan_opencpop} Choose vocoder type of svs task.
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
  --speech-stretchs     SPEECH_STRETCHS
                        The min and max values of the mel spectrum, using on diffusion of diffsinger.
```

`./local/synthesize_e2e.sh` 调用 `${BIN_DIR}/../synthesize_e2e.py`，即可从文本文件中合成波形。
`local/pinyin_to_phone.txt`来源于opencpop数据集中的README，表示opencpop中拼音到音素的映射。

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize_e2e.py [-h]
                         [--am {speedyspeech_csmsc,speedyspeech_aishell3,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech}]
                         [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                         [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                         [--speaker_dict SPEAKER_DICT] [--spk_id SPK_ID]
                         [--voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,style_melgan_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,wavernn_csmsc}]
                         [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                         [--voc_stat VOC_STAT] [--lang LANG]
                         [--inference_dir INFERENCE_DIR] [--ngpu NGPU]
                         [--text TEXT] [--output_dir OUTPUT_DIR]
                         [--pinyin_phone PINYIN_PHONE]
                         [--speech_stretchs SPEECH_STRETCHS]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {speedyspeech_csmsc,speedyspeech_aishell3,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech}
                        Choose acoustic model type of tts task.
       {diffsinger_opencpop} Choose acoustic model type of svs task.
  --am_config AM_CONFIG
                        Config of acoustic model.
  --am_ckpt AM_CKPT     Checkpoint file of acoustic model.
  --am_stat AM_STAT     mean and standard deviation used to normalize
                        spectrogram when training acoustic model.
  --phones_dict PHONES_DICT
                        phone vocabulary file.
  --speaker_dict SPEAKER_DICT
                        speaker id map file.
  --spk_id SPK_ID       spk id for multi speaker acoustic model
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,style_melgan_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,wavernn_csmsc}
                        Choose vocoder type of tts task.
        {pwgan_opencpop, hifigan_opencpop} Choose vocoder type of svs task.
  --voc_config VOC_CONFIG
                        Config of voc.
  --voc_ckpt VOC_CKPT   Checkpoint file of voc.
  --voc_stat VOC_STAT   mean and standard deviation used to normalize
                        spectrogram when training voc.
  --lang LANG           {zh, en, mix, canton} Choose language type of tts task.
                        {sing} Choose language type of svs task.
  --inference_dir INFERENCE_DIR
                        dir to save inference models
  --ngpu NGPU           if ngpu == 0, use cpu.
  --text TEXT           text to synthesize file, a 'utt_id sentence' pair per line for tts task.
                        A '{ utt_id input_type (is word) text notes note_durs}' or '{utt_id input_type (is phoneme) phones notes note_durs is_slurs}' pair per line for svs task.
  --output_dir OUTPUT_DIR
                        output dir.
  --pinyin_phone PINYIN_PHONE
                        pinyin to phone map file, using on sing_frontend.
  --speech_stretchs SPEECH_STRETCHS
                        The min and max values of the mel spectrum, using on diffusion of diffsinger.
```
1. `--am` 声学模型格式是否符合 {model_name}_{dataset}
2. `--am_config`, `--am_ckpt`, `--am_stat` 和 `--phones_dict` 是声学模型的参数，对应于 diffsinger 预训练模型中的 4 个文件。
3. `--voc` 声码器(vocoder)格式是否符合 {model_name}_{dataset}
4. `--voc_config`, `--voc_ckpt`, `--voc_stat` 是声码器的参数，对应于 parallel wavegan 预训练模型中的 3 个文件。
5. `--lang` tts对应模型的语言可以是 `zh`、`en`、`mix`和`canton`。 svs 对应的语言是 `sing` 。
6. `--test_metadata` 应为 `dump` 文件夹中 `test` 下的规范化元数据文件、
7. `--text` 是文本文件，其中包含要合成的句子。
8. `--output_dir` 是保存合成音频文件的目录。
9. `--ngpu` 要使用的GPU数，如果 ngpu==0，则使用 cpu。
10. `--inference_dir` 静态模型保存的目录。如果不加这一行，就不会生并保存成静态模型。
11. `--pinyin_phone` 拼音到音素的映射文件。
12. `--speech_stretchs` mel谱的最大最小值用于diffsinger中diffusion之前的线性拉伸。

注意： 目前 diffsinger 模型还不支持动转静，所以不要加 `--inference_dir`。


## 预训练模型
预先训练的 DiffSinger 模型：
- [diffsinger_opencpop_ckpt_1.4.0.zip](https://paddlespeech.bj.bcebos.com/t2s/svs/opencpop/diffsinger_opencpop_ckpt_1.4.0.zip)


DiffSinger 检查点包含下列文件。
```text
diffsinger_opencpop_ckpt_1.4.0.zip
├── default.yaml             # 用于训练 diffsinger 的默认配置
├── energy_stats.npy         # 训练 diffsinger 时如若需要 norm energy 会使用到的统计数据 
├── phone_id_map.txt         # 训练 diffsinger 时的音素词汇文件
├── pinyin_to_phone.txt      # 训练 diffsinger 时的拼音到音素映射文件
├── pitch_stats.npy          # 训练 diffsinger 时如若需要 norm pitch 会使用到的统计数据 
├── snapshot_iter_160000.pdz # 模型参数和优化器状态
├── speech_stats.npy         # 训练 diffsinger 时用于规范化频谱图的统计数据
└── speech_stretchs.npy      # 训练 diffusion 前用于 mel 谱拉伸的最小及最大值

```
您可以使用以下脚本通过使用预训练的 diffsinger 和 parallel wavegan 模型为 `${BIN_DIR}/../sentences_sing.txt` 合成句子
```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize_e2e.py \
  --am=diffsinger_opencpop \
  --am_config=diffsinger_opencpop_ckpt_1.4.0/default.yaml \
  --am_ckpt=diffsinger_opencpop_ckpt_1.4.0/snapshot_iter_160000.pdz \
  --am_stat=diffsinger_opencpop_ckpt_1.4.0/speech_stats.npy  \
  --voc=pwgan_opencpop \
  --voc_config=pwgan_opencpop_ckpt_1.4.0/default.yaml \
  --voc_ckpt=pwgan_opencpop_ckpt_1.4.0/snapshot_iter_100000.pdz \
  --voc_stat=pwgan_opencpop_ckpt_1.4.0/feats_stats.npy \
  --lang=sing \
  --text=${BIN_DIR}/../sentences_sing.txt \
  --output_dir=exp/default/test_e2e \
  --phones_dict=diffsinger_opencpop_ckpt_1.4.0/phone_id_map.txt \
  --pinyin_phone=diffsinger_opencpop_ckpt_1.4.0/pinyin_to_phone.txt \
  --speech_stretchs=diffsinger_opencpop_ckpt_1.4.0/speech_stretchs.npy
  
```
