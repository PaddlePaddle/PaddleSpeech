([简体中文](./README_cn.md)|English)
# Punctuation Restoration
## Introduction
Punctuation restoration is a common post-processing problem for Automatic Speech Recognition (ASR) systems. It is important to improve the readability of the transcribed text for the human reader and facilitate NLP tasks. 

This demo is an implementation to restore punctuation from raw text. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

You can choose one way from easy, meduim and hard to install paddlespeech.

### 2. Prepare Input
The input of this demo should be a text of the specific language that can be passed via argument.

### 3. Usage
- Command Line(Recommended)
  ```bash
  paddlespeech text --input 今天的天气真不错啊你下午有空吗我想约你一起去吃饭
  ```
  Usage:
  ```bash
  paddlespeech text --help
  ```
  Arguments:
  - `input`(required): Input raw text.
  - `task`: Choose subtask. Default: `punc`.
  - `model`: Model type of text task. Default: `ernie_linear_p7_wudao`.
  - `lang`: Choose model language.. Default: `zh`.
  - `config`: Config of text task. Use pretrained model when it is None. Default: `None`.
  - `ckpt_path`: Model checkpoint. Use pretrained model when it is None. Default: `None`.
  - `punc_vocab`: Vocabulary file of punctuation restoration task. Default: `None`.
  - `device`: Choose device to execute model inference. Default: default device of paddlepaddle in current environment.

  Output:
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
  Output:
  ```bash
  Text Result:
  今天的天气真不错啊！你下午有空吗？我想约你一起去吃饭。
  ```

### 4.Pretrained Models
Here is a list of pretrained models released by PaddleSpeech that can be used by command and python API:

- Punctuation Restoration
  | Model | Language | Number of Punctuation Characters
  | :--- | :---: | :---: 
  | ernie_linear_p3_wudao| zh | 3(，。？)
  | ernie_linear_p7_wudao| zh | 7(，。！？、：；)
