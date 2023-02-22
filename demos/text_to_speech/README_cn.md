(简体中文|[English](./README.md))

# 语音合成
## 介绍
语音合成是一种自然语言建模过程，其将文本转换为语音以进行音频演示。

这个 demo 是一个从给定文本生成音频的实现，它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。
## 使用方法
### 1. 安装
请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

你可以从 easy，medium，hard 三种方式中选择一种方式安装。

### 2. 准备输入

这个 demo 的输入是通过参数传递的特定语言的文本。
### 3. 使用方法
- 命令行 (推荐使用)
     默认的声学模型是 `Fastspeech2`，默认的声码器是 `HiFiGAN`，默认推理方式是动态图推理。
    - 中文
        ```bash
        paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！"
        ```
    - 批处理
        ```bash
        echo -e "1 欢迎光临。\n2 谢谢惠顾。" | paddlespeech tts
        ```
    - 中文，使用 `SpeedySpeech` 作为声学模型
        ```bash
        paddlespeech tts --am speedyspeech_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
        ```
    - 中文， 多说话人
    
        你可以改变 `spk_id`。
        ```bash
        paddlespeech tts --am fastspeech2_aishell3 --voc pwgan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0
        ```
    
     - 英文
        ```bash
        paddlespeech tts --am fastspeech2_ljspeech --voc pwgan_ljspeech --lang en --input "hello world"
        ```
    - 英文，多说话人
    
        你可以改变 `spk_id`。
        ```bash
        paddlespeech tts --am fastspeech2_vctk --voc pwgan_vctk --input "hello, boys" --lang en --spk_id 0
        ```
    - 中英文混合，多说话人
        你可以改变 `spk_id`。
        ```bash
        # The `am` must be `fastspeech2_mix`!
        # The `lang` must be `mix`!
        # The voc must be chinese datasets' voc now!
        # spk 174 is csmcc, spk 175 is ljspeech
        paddlespeech tts --am fastspeech2_mix --voc hifigan_csmsc --lang mix --input "热烈欢迎您在 Discussions 中提交问题，并在 Issues 中指出发现的 bug。此外，我们非常希望您参与到 Paddle Speech 的开发中！" --spk_id 174 --output mix_spk174.wav
        paddlespeech tts --am fastspeech2_mix --voc hifigan_aishell3 --lang mix --input "热烈欢迎您在 Discussions 中提交问题，并在 Issues 中指出发现的 bug。此外，我们非常希望您参与到 Paddle Speech 的开发中！" --spk_id 174 --output mix_spk174_aishell3.wav
        paddlespeech tts --am fastspeech2_mix --voc pwgan_csmsc --lang mix --input "我们的声学模型使用了 Fast Speech Two, 声码器使用了 Parallel Wave GAN and Hifi GAN." --spk_id 175 --output mix_spk175_pwgan.wav
        paddlespeech tts --am fastspeech2_mix --voc hifigan_csmsc --lang mix --input "我们的声学模型使用了 Fast Speech Two, 声码器使用了 Parallel Wave GAN and Hifi GAN." --spk_id 175 --output mix_spk175.wav
        ```
    - 中英文混合，单个男性说话人
        ```bash
        # male mix tts
        # The `lang` must be `mix`!
        paddlespeech tts --am fastspeech2_male --voc pwgan_male --lang mix --input "我们的声学模型使用了 Fast Speech Two, 声码器使用了 Parallel Wave GAN and Hifi GAN." --output male_mix_fs2_pwgan.wav
        paddlespeech tts --am fastspeech2_male --voc hifigan_male --lang mix --input "我们的声学模型使用了 Fast Speech Two, 声码器使用了 Parallel Wave GAN and Hifi GAN." --output male_mix_fs2_hifigan.wav
        ```
    - 使用 ONNXRuntime 推理：
        ```bash
        paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！" --output default.wav --use_onnx True
        paddlespeech tts --am speedyspeech_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！" --output ss.wav --use_onnx True
        paddlespeech tts --voc mb_melgan_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！" --output mb.wav --use_onnx True
        paddlespeech tts --voc pwgan_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！" --output pwgan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_aishell3 --voc pwgan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0 --output aishell3_fs2_pwgan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_aishell3 --voc hifigan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0 --output aishell3_fs2_hifigan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_ljspeech --voc pwgan_ljspeech --lang en --input "Life was like a box of chocolates, you never know what you're gonna get." --output lj_fs2_pwgan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_ljspeech --voc hifigan_ljspeech --lang en --input "Life was like a box of chocolates, you never know what you're gonna get." --output lj_fs2_hifigan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_vctk --voc pwgan_vctk --input "Life was like a box of chocolates, you never know what you're gonna get." --lang en --spk_id 0 --output vctk_fs2_pwgan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_vctk --voc hifigan_vctk --input "Life was like a box of chocolates, you never know what you're gonna get." --lang en --spk_id 0 --output vctk_fs2_hifigan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_male --voc pwgan_male --lang zh --input "你好，欢迎使用百度飞桨深度学习框架！" --output male_zh_fs2_pwgan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_male --voc pwgan_male --lang en --input "Life was like a box of chocolates, you never know what you're gonna get." --output male_en_fs2_pwgan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_male --voc pwgan_male --lang mix --input "热烈欢迎您在 Discussions 中提交问题，并在 Issues 中指出发现的 bug。此外，我们非常希望您参与到 Paddle Speech 的开发中！" --output male_fs2_pwgan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_male --voc hifigan_male --lang zh --input "你好，欢迎使用百度飞桨深度学习框架！" --output male_zh_fs2_hifigan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_male --voc hifigan_male --lang en --input "Life was like a box of chocolates, you never know what you're gonna get." --output male_en_fs2_hifigan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_mix --voc hifigan_male --lang mix --input "热烈欢迎您在 Discussions 中提交问题，并在 Issues 中指出发现的 bug。此外，我们非常希望您参与到 Paddle Speech 的开发中！" --output male_fs2_hifigan.wav --use_onnx True
        paddlespeech tts --am fastspeech2_mix --voc pwgan_csmsc --lang mix --spk_id 174 --input "热烈欢迎您在 Discussions 中提交问题，并在 Issues 中指出发现的 bug。此外，我们非常希望您参与到 Paddle Speech 的开发中！" --output mix_fs2_pwgan_csmsc_spk174.wav --use_onnx True
        ```

  使用方法：
  
  ```bash
  paddlespeech tts --help
  ```
  参数：
  - `input`(必须输入)：用于合成音频的文本。
  - `am`：TTS 任务的声学模型， 默认值：`fastspeech2_csmsc`。
  - `am_config`：声学模型的配置文件，若不设置则使用默认配置，默认值：`None`。
  - `am_ckpt`：声学模型的参数文件，若不设置则下载预训练模型使用，默认值：`None`。
  - `am_stat`：训练声学模型时用于正则化 mel 频谱图的均值标准差文件，默认值：`None`。
  - `phones_dict`：音素词表文件， 默认值：`None`。
  - `tones_dict`：声调词表文件， 默认值：`None`。
  - `speaker_dict`：说话人词表文件， 默认值：`None`。
  - `spk_id`：说话人 id， 默认值： `0`。
  - `voc`：TTS 任务的声码器， 默认值： `pwgan_csmsc`。
  - `voc_config`：声码器的配置文件，若不设置则使用默认配置，默认值：`None`。
  - `voc_ckpt`：声码器的参数文件，若不设置则下载预训练模型使用，默认值：`None`。
  - `voc_stat`：训练声码器时用于正则化 mel 频谱图的均值标准差文件，默认值：`None`。
  - `lang`：TTS 任务的语言， 默认值：`zh`。
  - `device`：执行预测的设备， 默认值：当前系统下 paddlepaddle 的默认 device。
  - `output`：输出音频的路径， 默认值：`output.wav`。
  - `use_onnx`: 是否使用 ONNXRuntime 进行推理。
  - `fs`: 使用特定 ONNX 模型时的采样率。

  输出：
  ```bash
  [2021-12-09 20:49:58,955] [    INFO] [log.py] [L57] - Wave file has been generated: output.wav
  ```

- Python API
     - 动态图推理:
        ```python
        import paddle
        from paddlespeech.cli.tts import TTSExecutor
        tts_executor = TTSExecutor()
        wav_file = tts_executor(
            text='今天的天气不错啊',
            output='output.wav',
            am='fastspeech2_csmsc',
            am_config=None,
            am_ckpt=None,
            am_stat=None,
            spk_id=0,
            phones_dict=None,
            tones_dict=None,
            speaker_dict=None,
            voc='pwgan_csmsc',
            voc_config=None,
            voc_ckpt=None,
            voc_stat=None,
            lang='zh',
            device=paddle.get_device())
        print('Wave file has been generated: {}'.format(wav_file))
        ```
    -  ONNXRuntime 推理:
        ```python
        from paddlespeech.cli.tts import TTSExecutor
        tts_executor = TTSExecutor()
        wav_file = tts_executor(
            text='对数据集进行预处理',
            output='output.wav',
            am='fastspeech2_csmsc',
            voc='hifigan_csmsc',
            lang='zh',
            use_onnx=True,
            cpu_threads=2)
        ```
 
  输出：
  ```bash
  Wave file has been generated: output.wav
  ```

### 4. 预训练模型
以下是 PaddleSpeech 提供的可以被命令行和 python API 使用的预训练模型列表：

- 声学模型
  | 模型 | 语言 |
  | :--- | :---: |
  |      speedyspeech_csmsc      |    zh    |
  |      fastspeech2_csmsc       |    zh    |
  |     fastspeech2_ljspeech     |    en    |
  |     fastspeech2_aishell3     |    zh    |
  |       fastspeech2_vctk       |    en    |
  | fastspeech2_cnndecoder_csmsc |    zh    |
  |       fastspeech2_mix        |   mix    |
  |       tacotron2_csmsc        |    zh    |
  |      tacotron2_ljspeech      |    en    |
  |       fastspeech2_male       |    zh    |
  |       fastspeech2_male       |    en    |
  |       fastspeech2_male       |   mix    |
  

- 声码器
  | 模型 | 语言 |
  | :--- | :---: |
  |         pwgan_csmsc          |    zh    |
  |        pwgan_ljspeech        |    en    |
  |        pwgan_aishell3        |    zh    |
  |          pwgan_vctk          |    en    |
  |       mb_melgan_csmsc        |    zh    |
  |      style_melgan_csmsc      |    zh    |
  |        hifigan_csmsc         |    zh    |
  |       hifigan_ljspeech       |    en    |
  |       hifigan_aishell3       |    zh    |
  |         hifigan_vctk         |    en    |
  |        wavernn_csmsc         |    zh    |
  |         pwgan_male           |    zh    |
  |        hifigan_male          |    zh    |
