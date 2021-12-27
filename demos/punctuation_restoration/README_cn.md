(简体中文|[English](./README.md))

# 标点恢复
## 介绍

标点恢复是语音识别系统中常见的后处理步骤。提高转录文本的可读性对于人类阅读和后续的自然语言处理任务是非常重要的。

这个 demo 是一个为原始文本恢复标点的实现，它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。

## 使用方法
### 1. 安装
请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

你可以从 easy，medium，hard 三中方式中选择一种方式安装。

### 2. 准备输入
这个 demo 的输入是通过参数传递的特定语言的文本。

### 3. 使用方法
- 命令行 (推荐使用)
  ```bash
  paddlespeech text --input 今天的天气真不错啊你下午有空吗我想约你一起去吃饭
  ```
  使用方法:
  ```bash
  paddlespeech text --help
  ```
  参数：
  - `input`(必须输入)：原始文本。
  - `task`：子任务，默认值：`punc`。
  - `model`：文本模型类型，默认值：`ernie_linear_p7_wudao`。
  - `lang`：模型语言， 默认值：`zh`。
  - `config`：文本任务的配置文件，若不设置则使用预训练模型中的默认配置，默认值：`None`。
  - `ckpt_path`：模型参数文件， 若不设置则下载预训练模型使用，默认值：`None`。
  - `punc_vocab`：标点恢复任务的标点词表文件，默认值：`None`。
  - `device`：执行预测的设备，默认值：当前系统下 paddlepaddle 的默认 device。

  输出：
  ```bash
  [2021-12-14 19:50:22,200] [    INFO] [log.py] [L57] - Text Result:
  今天的天气真不错啊！你下午有空吗？我想约你一起去吃饭。
  ```

- Python API
  ```python
  import paddle
  from paddlespeech.cli import TextExecutor

  text_executor = TextExecutor()
  result = text_executor(
      text='今天的天气真不错啊你下午有空吗我想约你一起去吃饭',
      task='punc',
      model='ernie_linear_p7_wudao',
      lang='zh',
      config=None,
      ckpt_path=None,
      punc_vocab=None,
      device=paddle.get_device())
  print('Text Result: \n{}'.format(result))
  ```
  输出：
  ```bash
  Text Result:
  今天的天气真不错啊！你下午有空吗？我想约你一起去吃饭。
  ```

### 预训练模型
以下是 PaddleSpeech 提供的可以被命令行和 python API 使用的预训练模型列表：

- 标点恢复
  | 模型 | 语言 | 标点类型数
  | :--- | :---: | :---: 
  | ernie_linear_p3_wudao| zh | 3(，。？)
  | ernie_linear_p7_wudao| zh | 7(，。！？、：；)
