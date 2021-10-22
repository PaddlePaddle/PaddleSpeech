# Speaker Encoder

本实验是的在多说话人数据集上以 Speaker Verification 为任务训练一个 speaker encoder, 这是作为 transfer learning from speaker verification to multispeaker text-to-speech synthesis 实验的一部分, 可以在 [tacotron2_aishell3](../tacotron2_aishell3) 中找到。用训练好的模型来提取音频的 utterance embedding.

## 模型

本实验使用的模型是 [GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION](https://arxiv.org/pdf/1710.10467.pdf) 中的 speaker encoder text independent 模型。使用的是 GE2E softmax 损失函数。

## 目录结构

```text
ge2e
├── README_cn.md
├── audio_processor.py
├── config.py
├── dataset_processors.py
├── inference.py
├── preprocess.py
├── random_cycle.py
├── speaker_verification_dataset.py
└── train.py
```

## 数据集下载

本实验支持了 Librispeech-other-500, VoxCeleb, VoxCeleb2,ai-datatang-200zh, magicdata 数据集。可以在对应的页面下载。

1. Librispeech/train-other-500

   英文多说话人数据集，[下载链接](https://www.openslr.org/resources/12/train-other-500.tar.gz)，我们的实验中仅用到了 train-other-500 这个子集。

2. VoxCeleb1

   英文多说话人数据集，[下载链接](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)，需要下载其中的 Audio Files 中的 Dev A 到 Dev D 四个压缩文件并合并解压。

3. VoxCeleb2

   英文多说话人数据集，[下载链接](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)，需要下载其中的 Audio Files 中的 Dev A 到 Dev H 八个压缩文件并合并解压。

4. Aidatatang-200zh

   中文多说话人数据集，[下载链接](https://www.openslr.org/62/)。

5. magicdata

   中文多说话人数据集，[下载链接](https://www.openslr.org/68/)。

如果用户需要使用其他的数据集，也可以自行下载并进行数据处理，只要符合如下的要求。

## 数据集预处理

训练中使用的数据集是多说话人数据集，transcription 并不会被使用。为了扩大数据的量，训练过程可以将多个数据集合并为一个。处理后的文件结果组织方式如下，每个句子的频谱存储为 `.npy` 格式。以 speaker-utterance 的两层目录结构存储。因为合并数据集的原因，为了避免 speaker id 冲突，dataset 名会被添加到 speaker id 前面。

```text
dataset_root
├── dataset01_speaker01/
│   ├── utterance01.npy
│   ├── utterance02.npy
│   └── utterance03.npy
├── dataset01_speaker02/
│   ├── utterance01.npy
│   ├── utterance02.npy
│   └── utterance03.npy
├── dataset02_speaker01/
│   ├── utterance01.npy
│   ├── utterance02.npy
│   └── utterance03.npy
└── dataset02_speaker02/
    ├── utterance01.npy
    ├── utterance02.npy
    └── utterance03.npy
```

运行数据处理脚本

```bash
python preprocess.py --datasets_root=<datasets_root> --output_dir=<output_dir> --dataset_names=<dataset_names>
```

其中 datasets_root 是包含多个原始数据集的路径，--output_dir 是多个数据集合并后输出的路径，dataset_names 是数据集的名称，多个数据集可以用逗号分割，比如 'librispeech_other, voxceleb1'. 目前支持的数据集有 librispeech_other, voxceleb1, voxceleb2, aidatatang_200zh, magicdata.

## 训练

数据处理完成后，使用如下的脚本训练。

```bash
python train.py --data=<data_path> --output=<output> --device="gpu" --nprocs=1
```

- `--data` 是处理后的数据集路径。
- `--output` 是训练结果的保存路径，一般使用 runs 下的一个子目录。保存结果包含 visualdl 的 log 文件，文本 log 记录，运行 config 备份，以及 checkpoints 目录，里面包含参数文件和优化器状态文件。如果指定的 output 路径包含此前的训练结果，训练前会自动加载最近的参数文件和优化器状态文件。
- `--device` 是运行设备，目前支持 'cpu' 和 'gpu'.
- `--nprocs` 是指定运行进程数。目前仅在使用 'gpu' 是支持多进程训练。可以配合 `CUDA_VISIBLE_DEVICES` 环境变量指定可见卡号。

另外还有几个选项。

- `--config` 是用于覆盖默认配置（默认配置可以查看 `config.py`) 的配置文件，为 `.yaml` 文件。
- `--opts` 是用命令行参数进一步覆盖配置。这是最后一个传入的命令行选项，用多组空格分隔的 KEY VALUE 对的方式传入。
- `--checkpoint_path` 指定从中恢复的 checkpoint, 不需要包含扩展名。同名的参数文件( `.pdparams`) 和优化器文件( `.pdopt`)会被加载以恢复训练。这个参数指定的恢复训练优先级高于自动从 `output` 文件夹中恢复训练。

## 预训练模型

预训练模型是在 Librispeech-other-500 和 voxceleb1 上训练到 1560k steps 后用 aidatatang_200h 和 magic_data 训练到 3000k 的结果。

下载链接 [ge2e_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/ge2e_ckpt_0.3.zip)

## 预测

使用训练好的模型进行预测，对一个数据集中的所有 utterance 生成一个 embedding.

```bash
python inference.py --input=<input> --output=<output> --checkpoint_path=<checkpoint_path> --device="gpu"
```

- `--input` 是需要处理的数据集的路径。
- `--output` 是处理的结果，它会保持和 `--input` 相同的文件夹结构，对应 input 中的每一个音频文件会有一个同名的 `*.npy` 文件，是从这个音频文件中提取到的 utterance embedding.
- `--checkpoint_path` 为用于预测的参数文件路径，不包含扩展名。
- `--pattern` 是用于筛选数据集中需要处理的音频文件的通配符模式，默认为 `*.wav`.
- `--device` 和 `--opts` 的语义和训练脚本一致。

## 参考文献

1. [GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION](https://arxiv.org/pdf/1710.10467.pdf)
2. [Transfer Learning from Speaker Verification toMultispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf)
