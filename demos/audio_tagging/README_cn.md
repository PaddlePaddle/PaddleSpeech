
(简体中文|[English](./README.md))

# 声音分类
## 介绍
声音分类任务为音频片段添加一个或多个标签的任务，包括音乐分类、声学场景分类、音频事件分类等。

这个 demo 使用 527 个 [AudioSet](https://research.google.com/audioset/) 数据集中的标签为音频进行分类，它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。

## 使用方法
### 1. 安装
请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

你可以从 easy，medium，hard 三中方式中选择一种方式安装。

### 2. 准备输入
这个 demo 的输入应该是一个 WAV 文件（`.wav`），

可以下载此 demo 的示例音频：
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav
```

### 3. 使用方法
- 命令行 (推荐使用)
  ```bash
  paddlespeech cls --input ./cat.wav --topk 10
  ```
  使用方法：
  ```bash
  paddlespeech cls --help
  ```
  参数：
  - `input`(必须输入)： 用于分类的音频。
  - `model`： 声音分类任务的模型， 默认值： `panns_cnn14`.
  - `config`： 声音分类任务的配置文件，若不设置则使用预训练模型中的默认配置，  默认值： `None`。
  - `ckpt_path`：模型参数文件， 若不设置则下载预训练模型使用， 默认值： `None`。
  - `label_file`：声音分类任务的标签文件，若不是设置则使用音频数据集标签，默认值： `None`。
  - `topk`：展示分类结果的 topk 个结果，默认值： `1`。
  - `device`：执行预测的设备，默认值：当前系统下 paddlepaddle 的默认 device。

  输出：
  ```bash
  [2021-12-08 14:49:40,671] [    INFO] [utils.py] [L225] - CLS Result:
  Cat: 0.8991316556930542
  Domestic animals, pets: 0.8806838393211365
  Meow: 0.8784668445587158
  Animal: 0.8776564598083496
  Caterwaul: 0.2232048511505127
  Speech: 0.03101264126598835
  Music: 0.02870696596801281
  Inside, small room: 0.016673989593982697
  Purr: 0.008387474343180656
  Bird: 0.006304860580712557
  ```

- Python API
  ```python
  import paddle
  from paddlespeech.cli import CLSExecutor

  cls_executor = CLSExecutor()
  result = cls_executor(
      model='panns_cnn14',
      config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
      label_file=None,
      ckpt_path=None,
      audio_file='./cat.wav',
      topk=10,
      device=paddle.get_device())
  print('CLS Result: \n{}'.format(result))
  ```
  输出：
  ```bash
  CLS Result:
  Cat: 0.8991316556930542
  Domestic animals, pets: 0.8806838393211365
  Meow: 0.8784668445587158
  Animal: 0.8776564598083496
  Caterwaul: 0.2232048511505127
  Speech: 0.03101264126598835
  Music: 0.02870696596801281
  Inside, small room: 0.016673989593982697
  Purr: 0.008387474343180656
  Bird: 0.006304860580712557
  ```

### 4. 预训练模型

以下是 PaddleSpeech 提供的可以被命令行和 python api 使用的预训练模型列表：

| 模型 | 采样率
| :--- | :---: 
| panns_cnn6| 32000
| panns_cnn10| 32000
| panns_cnn14| 32000
