# 语音识别: DeepSpeech2

*语音识别: DeepSpeech2*是一个采用[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)平台的端到端自动语音识别（ASR）引擎的开源项目，具体原理参考这篇论文[Baidu's Deep Speech 2 paper](http://proceedings.mlr.press/v48/amodei16.pdf)。
我们的愿景是为语音识别在工业应用和学术研究上，提供易于使用、高效和可扩展的工具，包括训练，推理，测试模块，以及分布式的[PaddleCloud](https://github.com/PaddlePaddle/cloud)训练和demo部署。同时，我们还将发布一些预训练好的英语和普通话模型。

## 目录
- [安装](#安装)
- [开始](#开始)
- [数据准备](#数据准备)
- [训练模型](#训练模型)
- [数据增强管道](#数据增强管道)
- [推断和评价](#推断和评价)
- [在Docker容器上运行](#在Docker容器上运行)
- [分布式云训练](#分布式云训练)
- [超参数调整](#超参数调整)
- [训练汉语语言](#训练汉语语言)
- [用自己的声音尝试现场演示](#用自己的声音尝试现场演示)
- [发布模型](#发布模型)
- [试验和基准](#试验和基准)
- [问题和帮助](#问题和帮助)

## 安装
为了避免环境配置问题，强烈建议在[Docker容器上运行](#在Docker容器上运行)，另外请按照下面的指南安装依赖项。

### 前提
- 只支持Python 2.7
- PaddlePaddle最新版本(请参考[安装指南](https://github.com/PaddlePaddle/Paddle#installation))

### 安装
- 请确保以下库或工具已安装完毕：`pkg-config`, `flac`, `ogg`, `vorbis`, `boost` 和 `swig`, 以上可以通过`apt-get`安装：

```bash
sudo apt-get install -y pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig
```

- 为剩下的依赖项运行安装脚本

```bash
git clone https://github.com/PaddlePaddle/DeepSpeech.git
cd DeepSpeech
sh setup.sh
```

## 开始

`./examples`里的一些shell脚本将帮助我们在一些公开数据集(比如：[LibriSpeech](http://www.openslr.org/12/), [Aishell](http://www.openslr.org/33)) 进行快速尝试，包括了数据准备，模型训练，案例推断和模型评价。阅读这些例子将帮助你理解如何应用你的数据集。

`./examples`目录中的一些脚本配置使用了8个GPU。如果你没有8个可用的GPU，请修改`CUDA_VISIBLE_DEVICES`和`--trainer_count`。如果你没有可用的GPU，请设置`--use_gpu`为False，这样程序会用CPU代替GPU。另外如果发生内存不足的问题，减小`--batch_size`即可。

让我们先看看[LibriSpeech dataset](http://www.openslr.org/12/)小样本集的例子。

- 转到目录

    ```bash
    cd examples/tiny
    ```

    注意这仅仅是LibriSpeech一个小数据集的例子。如果你想尝试完整的数据集（可能需要花好几天来训练模型），请使用这个路径`examples/librispeech`。   
- 准备数据

    ```bash
    sh run_data.sh
    ```

    运行`run_data.sh`脚本将会下载数据集，产出manifests文件，收集一些归一化需要的统计信息并建立词表。当数据准备完成之后，下载完的数据（仅有LibriSpeech一部分）在`~/.cache/paddle/dataset/speech/libri`中；其对应的manifest文件，均值标准差和词表文件在`./data/tiny`中。在第一次执行的时候一定要执行这个脚本，在接下来所有的实验中我们都会用到这个数据集。    
- 训练你自己的ASR模型

    ```bash
    sh run_train.sh
    ```

    `run_train.sh`将会启动训练任务，训练日志会打印到stdout，并且模型每个时期(epoch)的检查点都会保存到`./checkpoints/tiny`目录中。这些检查点可以用来恢复训练，推断，评价和部署。
- 用已有的模型进行案例推断

    ```bash
    sh run_infer.sh
    ```

    `run_infer.sh`将会利用训完的模型展现一些（默认10个）样本语音到文本的解码结果。由于当前模型只使用了LibriSpeech一部分数据集训练，因此性能可能不会太好。为了看到更好模型上的表现，你可以下载一个已训练好的模型（用完整的LibriSpeech训练了好几天）来做推断。

    ```bash
    sh run_infer_golden.sh
    ```
- 评价一个已经存在的模型

    ```bash
    sh run_test.sh
    ```

    `run_test.sh`能够利用误字率（或字符错误率）来评价模型。类似的，你可以下载一个完全训练好的模型来测试它的性能：

    ```bash
    sh run_test_golden.sh
    ```

更多细节会在接下来的章节中阐述。祝你在*语音识别: DeepSpeech2*ASR引擎学习中过得愉快！


## 数据准备

### 生成Manifest

*语音识别: DeepSpeech2*接受文本**manifest**文件作为数据接口。manifest文件包含了一系列语音数据，其中每一行代表一个json格式的音频元数据（比如文件路径，描述，时长）。具体格式如下：

```
{"audio_filepath": "/home/work/.cache/paddle/Libri/134686/1089-134686-0001.flac", "duration": 3.275, "text": "stuff it into you his belly counselled him"}
{"audio_filepath": "/home/work/.cache/paddle/Libri/134686/1089-134686-0007.flac", "duration": 4.275, "text": "a cold lucid indifference reigned in his soul"}
```

如果你要使用自定义数据，你只需要按照以上格式生成自己的manifest文件即可。训练，推断以及其他所有模块都能够根据manifest文件获取到音频数据，包括他们的元数据。

关于如何生成manifest文件，请参考`data/librispeech/librispeech.py`。该脚本将会下载LibriSpeech数据集并生成manifest文件。

### 计算均值和标准差用于归一化

为了对音频特征进行z-score归一化（零均值，单位标准差），我们必须预估一些训练样本特征的均值和标准差：

```bash
python tools/compute_mean_std.py \
--num_samples 2000 \
--specgram_type linear \
--manifest_paths data/librispeech/manifest.train \
--output_path data/librispeech/mean_std.npz
```

以上这段代码会计算在`data/librispeech/manifest.train`路径中，2000个随机采样音频剪辑的功率谱特征均值和标准差，并将结果保存在`data/librispeech/mean_std.npz`中，方便以后使用。

### 建立词表

转换录音为索引用于训练，解码，再将一系列索引转换为文本等操作需要一个可能会出现字符集合的词表。`tools/build_vocab.py`脚本将生成这种基于字符的词表。

```bash
python tools/build_vocab.py \
--count_threshold 0 \
--vocab_path data/librispeech/eng_vocab.txt \
--manifest_paths data/librispeech/manifest.train
```

他将`data/librispeech/manifest.train`目录中的所有录音文本写入词表文件`data/librispeeech/eng_vocab.txt`，并且没有词汇截断(`--count_threshold 0`)。

### 更多帮助

获得更多帮助：

```bash
python data/librispeech/librispeech.py --help
python tools/compute_mean_std.py --help
python tools/build_vocab.py --help
```

## 训练模型

`train.py`是训练模块的主要调用者。使用示例如下。

- 开始使用8片GPU训练：

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --trainer_count 8
    ```

- 开始使用16片GPU训练：

    ```
    python train.py --use_gpu False --trainer_count 16
    ```
    
- 从检查点恢复训练：

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python train.py \
    --init_model_path CHECKPOINT_PATH_TO_RESUME_FROM
    ```

获得更多帮助：

```bash
python train.py --help
```
或参考 `example/librispeech/run_train.sh`.

## 数据增强管道

数据增强是用来提升深度学习性能的非常有效的技术。我们通过在原始音频中添加小随机扰动（标签不变转换）获得新音频来增强我们的语音数据。你不必自己合成，因为数据增强已经嵌入到数据提供者中，能在训练模型时每个epoch中随机的合成音频。

目前提供六个可选的增强组件供选择，配置并插入处理流水线。

  - 音量扰动
  - 速度扰动
  - 移动扰动
  - 在线贝叶斯归一化
  - 噪声干扰（需要背景噪音的音频文件）
  - 脉冲响应（需要脉冲音频文件）

为了让训练模块知道需要哪些增强组件以及它们的处理顺序，我们需要事先准备一个[JSON](http://www.json.org/)格式的*扩展配置文件*。例如：

```
[{
    "type": "speed",
    "params": {"min_speed_rate": 0.95,
               "max_speed_rate": 1.05},
    "prob": 0.6
},
{
    "type": "shift",
    "params": {"min_shift_ms": -5,
               "max_shift_ms": 5},
    "prob": 0.8
}]
```

当`trainer.py`的`--augment_conf_file`参数被设置为上述示例配置文件的路径时，每个epoch中的每个音频片段都将被处理。首先，均匀随机采样速率会有60％的概率在0.95和1.05之间对音频片段进行速度扰动。然后，音频片段有80％的概率在时间上被挪移，挪移偏差值是-5毫秒和5毫秒之间的随机采样。最后，这个新合成的音频片段将被传送给特征提取器，以用于接下来的训练。

有关其他配置实例，请参考`conf/augmenatation.config.example`.

使用数据增强技术时要小心，由于扩大了训练和测试集的差异，不恰当的增强会对训练模型不利。

## 推断和评价

### 准备语言模型

提升解码器的性能需要准备语言模型。我们准备了两种语言模型（有损压缩）供用户下载和尝试。一个是英语模型，另一个是普通话模型。用户可以执行以下命令来下载已经训练好的语言模型：

```bash
cd models/lm
sh download_lm_en.sh
sh download_lm_ch.sh
```

如果你想训练自己更好的语言模型，请参考[KenLM]（https://github.com/kpu/kenlm）获取教程。在这里，我们提供一些技巧来展示我们如何准备我们的英语和普通话模型。开始训练的时候，你可以参考这些技巧。


#### 英语语言模型

英语语料库来自[Common Crawl Repository](http://commoncrawl.org)，您可以从[statmt](http://data.statmt.org/ngrams/deduped_en)下载它。我们使用en.00部分来训练我们的英语语言模型。训练前有一些预处理步骤如下：
  
  * 不在\[A-Za-z0-9\s'\]（\s表示空白字符）中的字符将被删除，阿拉伯数字被转换为英文数字，比如“1000”转换为one thousand。
  * 重复的空白字符被压缩为一个，并且开始的空白字符将被删除。请注意，所有的录音都是小写字母，因此所有字符都转换为小写字母。
  * 选择前40万个最常用的单词来建立词表，其余部分将被替换为“UNKNOWNWORD”。

现在预处理完成了，我们得到一个干净的语料库来训练语言模型。我们发布的语言模型版本使用了参数“-o 5 --prune 0 1 1 1 1”来训练。“-o 5”表示语言模型的最大order为5。“--prune 0 1 1 1 1”表示每个order的计数阈值，更具体地说，它将第2个以及更高的order修剪为单个。为了节省磁盘存储空间，我们将使用参数“-a 22 -q 8 -b 8”将arpa文件转换为“trie”二进制文件。“-a”表示在“trie”中用于切分的指针的最高位数。“-q -b”是概率和退避的量化参数。

#### 普通话语言模型

与英语语言模型不同的是，普通话语言模型是基于字符的，其中每一位都是中文汉字。我们使用内部语料库来训练发布的汉语语言模型。该语料库包含数十亿汉字。预处理阶段与英语语言模型差别很小，主要步骤包括：

  * 删除开始和结尾的空白字符。
  * 删除英文标点和中文标点。
  * 在两个字符之间插入空白字符。

请注意，发布的语言模型只包含中文简体字。预处理完成后，我们开始训练语言模型。这个小的语言模型训练关键参数是“-o 5 --prune 0 1 2 4 4”，“-o 5”是针对大语言模型。请参考上面的部分了解每个参数的含义。我们还使用默认设置将arpa文件转换为二进制文件。

### 语音到文本推断

推断模块调用者为`infer.py`，可以用来推断，解码，以及给一些给定音频剪辑进行可视化语音到文本的结果。这有助于对ASR模型的性能进行直观和定性的评估。

- GPU版本的推断：

    ```bash
    CUDA_VISIBLE_DEVICES=0 python infer.py --trainer_count 1
    ```

- CPU版本的推断：

    ```bash
    python infer.py --use_gpu False --trainer_count 12
    ```

我们提供两种类型的CTC解码器：*CTC贪心解码器*和*CTC波束搜索解码器*。*CTC贪心解码器*是简单的最佳路径解码算法的实现，在每个时间步选择最可能的字符，因此是贪心的并且是局部最优的。[*CTC波束搜索解码器*](https://arxiv.org/abs/1408.2873)另外使用了启发式广度优先图搜索以达到近似全局最优; 它也需要预先训练的KenLM语言模型以获得更好的评分和排名。解码器类型可以用参数`--decoding_method`设置。

获得更多帮助：

```
python infer.py --help
```
或参考`example/librispeech/run_infer.sh`.

### 评估模型

要定量评估模型的性能，请运行：

- 带GPU版本评估

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --trainer_count 8
    ```

- CPU版本评估

    ```bash
    python test.py --use_gpu False --trainer_count 12
    ```

错误率（默认：误字率;可以用--error_rate_type设置）将被打印出来。

获得更多帮助：

```bash
python test.py --help
```
或参考`example/librispeech/run_test.sh`.

## 超参数调整

[*CTC波束搜索解码器*](https://arxiv.org/abs/1408.2873)的超参数$\alpha$（语言模型权重）和$\beta$（单词插入权重）对解码器的性能有非常显著的影响。当声学模型更新时，最好在验证集上重新调整它们。

`tools/tune.py`会进行2维网格查找超参数$\alpha$和$\beta$。您必须提供$\alpha$和$\beta$的范围，以及尝试的次数。

- 带GPU版的调整：

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python tools/tune.py \
    --trainer_count 8 \
    --alpha_from 1.0 \
    --alpha_to 3.2 \
    --num_alphas 45 \
    --beta_from 0.1 \
    --beta_to 0.45 \
    --num_betas 8
    ```

- CPU版的调整：

    ```bash
    python tools/tune.py --use_gpu False
    ```
网格搜索将会在超参数空间的每个点处打印出WER(误字率)或者CER(字符错误率)，并且可选择绘出误差曲面。合适的超参数范围应包括WER/CER误差表面的全局最小值，如下图所示。

<p align="center">
<img src="docs/images/tuning_error_surface.png" width=550>
<br/>调整LibriSpeech的dev-clean集合的误差曲面示例
</p>

通常，如图所示，语言模型权重（$\alpha$）的变化显著影响CTC波束搜索解码器的性能。更好的方法是首先调整多批数据（可指定数量）以找出适当的超参数范围，然后更改为整个验证集以进行精确调整。

调整之后，您可以在推理和评价模块中重置$\alpha$和$\beta$，以检查它们是否真的有助于提高ASR性能。更多帮助如下：

```bash
python tune.py --help
```
或参考`example/librispeech/run_tune.sh`.

## 在Docker容器上运行

Docker是一个开源工具，用于在孤立的环境中构建，发布和运行分布式应用程序。此项目的Docker镜像已在[hub.docker.com](https://hub.docker.com)中提供，并安装了所有依赖项，其中包括预先构建的PaddlePaddle，CTC解码器以及其他必要的Python和第三方库。这个Docker映像需要NVIDIA GPU的支持，所以请确保它的可用性并已完成[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)的安装。

采取以下步骤来启动Docker镜像：

- 下载Docker镜像

```bash
nvidia-docker pull paddlepaddle/deep_speech:latest-gpu
```

- git clone这个资源库

```
git clone https://github.com/PaddlePaddle/DeepSpeech.git
```

- 运行Docker镜像

```bash
sudo nvidia-docker run -it -v $(pwd)/DeepSpeech:/DeepSpeech paddlepaddle/deep_speech:latest-gpu /bin/bash
```

现在返回并从[开始](#开始)部分开始，您可以在Docker容器中同样执行模型训练，推断和超参数调整。

## 分布式云训练

我们还为用户提供云训练模块[PaddleCloud](https://github.com/PaddlePaddle/cloud)以便用户进行集群训练，利用多台机器达到更快的训练速度。首先，请按照[PaddleCloud用法](https://github.com/PaddlePaddle/cloud/blob/develop/doc/usage_cn.md#%E4%B8%8B%E8%BD%BD%E5%B9%B6%E9%85%8D%E7%BD%AEpaddlecloud)安装PaddleCloud客户端并注册PaddleCloud账户。

请按照以下步骤提交训练任务：

- 转到目录：

    ```bash
    cd cloud
    ```
- 上传数据：

    数据必须上传到PaddleCloud文件系统才能在云作业中访问。`pcloud_upload_data.sh`负责进行数据打包和上传：

    ```bash
    sh pcloud_upload_data.sh
    ```

    给定manifest文件，`pcloud_upload_data.sh`会进行以下处理：

    - 提取输入清单中列出的音频文件。
    - 将它们打包成指定数量的tar文件。
    - 将这些tar文件上传到PaddleCloud文件系统。
    - 通过用PaddleCloud文件系统路径替换本地文件系统路径来创建云manifest文件。云作业将通过新的manifest文件获取到音频文件的位置及其元信息。

    对于云训练模型来说以上步骤只需做一次。之后这些数据会在云文件系统上保持不变，并可在之后的任务中反复使用。

    有关参数的详细信息，请参考[在PaddleCloud上训练DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/cloud)。

 - 配置训练参数

    在`pcloud_submit.sh`中配置云任务参数（例如`NUM_NODES`，`NUM_GPUS`，`CLOUD_TRAIN_DIR`，`JOB_NAME`等），然后在`pcloud_train.sh`中配置其他的超参数训练（和本地训练一样）。

    有关参数的详细信息，请参阅[在PaddleCloud上训练DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/cloud)。


 - 提交任务

    运行：

    ```bash
    sh pcloud_submit.sh
    ```
    一个训练任务已经提交给PaddleCloud，并将任务名输出到控制台。

  - 获取训练日志
    
    执行以下命令以列出你提交的所有任务以及它们的运行状态：

    ```bash
    paddlecloud get jobs
    ```

    运行此操作，将打印相应的任务日志。
    
    ```bash
    paddlecloud logs -n 10000 $REPLACED_WITH_YOUR_ACTUAL_JOB_NAME
    ```

有关PaddleCloud用法的更多信息，请参阅[PaddleCloud用法](https://github.com/PaddlePaddle/cloud/blob/develop/doc/usage_cn.md#提交任务)。

有关PaddleCloud的DeepSpeech2训练的更多信息，请参阅
[Train DeepSpeech2 on PaddleCloud](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/cloud).

## 训练普通话语言

普通话语言训练与英语训练的关键步骤相同，我们提供了一个```examples/aishell```中Aishell的普通话训练例子。如上所述，请执行```sh run_data.sh```, ```sh run_train.sh```, ```sh run_test.sh```和```sh run_infer.sh```做相应的数据准备，训练，测试和推断。我们还准备了一个预训练过的模型（执行./models/aishell/download_model.sh下载）供用户使用```run_infer_golden.sh```和```run_test_golden.sh```来。请注意，与英语语言模型不同，普通话语言模型是基于汉字的，请运行```tools/tune.py```来查找最佳设置。

##用自己的声音尝试现场演示

到目前为止，一个ASR模型已经训练完毕，并且进行了定性测试（`infer.py`）和用现有的音频文件进行定量测试（`test.py`）。但目前还没有用你自己的声音进行测试。`deploy/demo_server.py`和`deploy/demo_client.py`能够快速构建一个利用训完的模型，对ASR引擎进行实时演示系统，使你能够用自己的语音测试和演示。

要启动演示服务，请在控制台中运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python deploy/demo_server.py \
--trainer_count 1 \
--host_ip localhost \
--host_port 8086
```

对于运行demo客户端的机器（可能不是同一台机器），请在继续之前执行以下安装。

比如，对于MAC OS X机器：

```bash
brew install portaudio
pip install pyaudio
pip install pynput
```

然后启动客户端，请在另一个控制台中运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python -u deploy/demo_client.py \
--host_ip 'localhost' \
--host_port 8086
```

现在，在客户端控制台中，按下`whitespace`键，按住并开始讲话。讲话完毕请释放该键以让控制台中显示的语音到文本结果。要退出客户端，只需按`ESC`键。

请注意，`deploy/demo_client.py`必须在带麦克风设备的机器上运行，而`deploy/demo_server.py`可以在没有任何录音硬件的情况下运行，例如任何远程服务器机器。如果服务器和客户端使用两台独立的机器运行，只需要注意将`host_ip`和`host_port`参数设置为实际可访问的IP地址和端口。如果它们在单台机器上运行，则不用作任何处理。

请参考`examples/mandarin/run_demo_server.sh`，它将首先下载一个预先训练过的普通话模型（用3000小时的内部语音数据训练），然后用模型启动演示服务器。通过运行`examples/mandarin/run_demo_client.sh`，你可以说普通话来测试它。如果您想尝试其他模型，只需更新脚本中的`--model_path`参数即可。

获得更多帮助：

```bash
python deploy/demo_server.py --help
python deploy/demo_client.py --help
```

## 发布模型

#### 语音模型发布

语种  | 模型名 | 训练数据 | 语音时长
:-----------: | :------------: | :----------: |  -------:
English  | [LibriSpeech Model](http://cloud.dlnel.org/filepub/?uuid=117cde63-cd59-4948-8b80-df782555f7d6) | [LibriSpeech Dataset](http://www.openslr.org/12/) | 960 h
English  | [BaiduEN8k Model](http://cloud.dlnel.org/filepub/?uuid=37a1c211-ec47-494c-973c-31437a10ae90) | Baidu Internal English Dataset | 8628 h
Mandarin | [Aishell Model](http://cloud.dlnel.org/filepub/?uuid=61de63b9-6904-4809-ad95-0cc5104ab973) | [Aishell Dataset](http://www.openslr.org/33/) | 151 h
Mandarin | [BaiduCN1.2k Model](http://cloud.dlnel.org/filepub/?uuid=499569a6-0025-4f40-83e6-1c99527431a6) | Baidu Internal Mandarin Dataset | 1204 h

#### 语言模型发布

语言模型 | 训练数据 | 基于的字符 | 大小 | 描述
:-------------:| :------------:| :-----: | -----: | :-----------------
[English LM](http://paddlepaddle.bj.bcebos.com/model_zoo/speech/common_crawl_00.prune01111.trie.klm) |  [CommonCrawl(en.00)](http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.00.deduped.xz) | Word-based | 8.3 GB | Pruned with 0 1 1 1 1; <br/> About 1.85 billion n-grams; <br/> 'trie'  binary with '-a 22 -q 8 -b 8'
[Mandarin LM Small](http://cloud.dlnel.org/filepub/?uuid=d21861e4-4ed6-45bb-ad8e-ae417a43195e) | Baidu Internal Corpus | Char-based | 2.8 GB | Pruned with 0 1 2 4 4; <br/> About 0.13 billion n-grams; <br/> 'probing' binary with default settings
[Mandarin LM Large](http://cloud.dlnel.org/filepub/?uuid=245d02bb-cd01-4ebe-b079-b97be864ec37) | Baidu Internal Corpus | Char-based | 70.4 GB | No Pruning; <br/> About 3.7 billion n-grams; <br/> 'probing' binary with default settings

## 实验和基准

#### 英语模型的基准测试结果（字错误率）

测试集                | LibriSpeech Model | BaiduEN8K Model
:---------------------  | ---------------:  | -------------------:
LibriSpeech Test-Clean  |   6.85            |   5.41
LibriSpeech Test-Other  |   21.18           |   13.85
VoxForge American-Canadian | 12.12          |   7.13
VoxForge Commonwealth   |   19.82           |   14.93
VoxForge European       |   30.15           |   18.64
VoxForge Indian         |   53.73           |   25.51
Baidu Internal Testset  |   40.75           |   8.48

为了在VoxForge数据上重现基准测试结果，我们提供了一个脚本来下载数据并生成VoxForge方言manifest文件。请到```data/voxforge```执行````run_data.sh```来获取VoxForge方言manifest文件。请注意，VoxForge数据可能会持续更新，生成的清单文件可能与我们评估的清单文件有所不同。


#### 普通话模型的基准测试结果（字符错误率）

测试集                |  BaiduCN1.2k Model
:---------------------  |  -------------------:
Baidu Internal Testset  |   12.64

#### 多GPU加速

我们对1,2,4,8,16个Tesla K40m GPU的训练时间（LibriSpeech样本的子集，其音频持续时间介于6.0和7.0秒之间）进行比较。它表明，已经实现了具有多个GPU的**近线性**加速。在下图中，训练的时间（以秒为单位）显示在蓝色条上。

<img src="docs/images/multi_gpu_speedup.png" width=450><br/>

| # of GPU  | 加速比 |
| --------  | --------------:   |
| 1         | 1.00 X |
| 2         | 1.97 X |
| 4         | 3.74 X |
| 8         | 6.21 X |
|16         | 10.70 X |

`tools/profile.sh`提供了上述分析工具.

## 问题和帮助

欢迎您在[Github问题](https://github.com/PaddlePaddle/models/issues)中提交问题和bug。也欢迎您为这个项目做出贡献。

