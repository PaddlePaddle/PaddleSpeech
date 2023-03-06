([简体中文](./README_cn.md)|English)
# TTS (Text To Speech)

## Introduction
Text-to-speech (TTS) is a natural language modeling process that requires changing units of text into units of speech for audio presentation. 

This demo is an implementation to generate audio from the given text. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

You can choose one way from easy, meduim and hard to install paddlespeech.

### 2. Prepare Input
The input of this demo should be a text of the specific language that can be passed via argument.
### 3. Usage
- Command Line (Recommended)
    The default acoustic model is `Fastspeech2`, and the default vocoder is `HiFiGAN`, the default inference method is dygraph inference. 
    - Chinese
        ```bash
        paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！"
        ```
    - Batch Process
        ```bash
        echo -e "1 欢迎光临。\n2 谢谢惠顾。" | paddlespeech tts
        ```
    - Chinese, use `SpeedySpeech` as the acoustic model
        ```bash
        paddlespeech tts --am speedyspeech_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
        ```
    - Chinese, multi-speaker
    
        You can change `spk_id` here.
        ```bash
        paddlespeech tts --am fastspeech2_aishell3 --voc pwgan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0
        ```
    
     - English
        ```bash
        paddlespeech tts --am fastspeech2_ljspeech --voc pwgan_ljspeech --lang en --input "hello world"
        ```
    - English, multi-speaker
    
        You can change `spk_id` here.
        ```bash
        paddlespeech tts --am fastspeech2_vctk --voc pwgan_vctk --input "hello, boys" --lang en --spk_id 0
        ```
    - Chinese English Mixed, multi-speaker
        You can change `spk_id` here.
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
    - Chinese English Mixed, single male spk
        ```bash
        # male mix tts
        # The `lang` must be `mix`!
        paddlespeech tts --am fastspeech2_male --voc pwgan_male --lang mix --input "我们的声学模型使用了 Fast Speech Two, 声码器使用了 Parallel Wave GAN and Hifi GAN." --output male_mix_fs2_pwgan.wav
        paddlespeech tts --am fastspeech2_male --voc hifigan_male --lang mix --input "我们的声学模型使用了 Fast Speech Two, 声码器使用了 Parallel Wave GAN and Hifi GAN." --output male_mix_fs2_hifigan.wav
        ```
    - Cantonese
        ```bash
        paddlespeech tts --am fastspeech2_canton --voc pwgan_aishell3 --input "各个国家有各个国家嘅国歌" --lang canton --spk_id 10
        ```
    - Use ONNXRuntime infer：
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
        paddlespeech tts --am fastspeech2_canton --voc pwgan_aishell3 --lang canton --spk_id 10 --input "各个国家有各个国家嘅国歌" --output output_canton.wav --use_onnx True
        ```

  Usage:
  
  ```bash
  paddlespeech tts --help
  ```
  Arguments:
  - `input`(required): Input text to generate..
  - `am`: Acoustic model type of tts task. Default: `fastspeech2_csmsc`.
  - `am_config`: Config of acoustic model. Use deault config when it is None. Default: `None`.
  - `am_ckpt`: Acoustic model checkpoint. Use pretrained model when it is None. Default: `None`.
  - `am_stat`: Mean and standard deviation used to normalize spectrogram when training acoustic model. Default: `None`.
  - `phones_dict`: Phone vocabulary file. Default: `None`.
  - `tones_dict`: Tone vocabulary file. Default: `None`.
  - `speaker_dict`: speaker id map file. Default: `None`.
  - `spk_id`: Speaker id for multi speaker acoustic model. Default: `0`.
  - `voc`: Vocoder type of tts task. Default: `pwgan_csmsc`.
  - `voc_config`: Config of vocoder. Use deault config when it is None. Default: `None`.
  - `voc_ckpt`: Vocoder checkpoint. Use pretrained model when it is None. Default: `None`.
  - `voc_stat`: Mean and standard deviation used to normalize spectrogram when training vocoder. Default: `None`.
  - `lang`: Language of tts task. Default: `zh`.
  - `device`: Choose device to execute model inference. Default: default device of paddlepaddle in current environment.
  - `output`: Output wave filepath. Default: `output.wav`.
  - `use_onnx`: whether to usen ONNXRuntime inference.
  - `fs`: sample rate for ONNX models when use specified model files.

  Output:
  ```bash
  [2021-12-09 20:49:58,955] [    INFO] [log.py] [L57] - Wave file has been generated: output.wav
  ```

- Python API
    - Dygraph infer:
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
    - ONNXRuntime infer:
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
 
  Output:
  ```bash
  Wave file has been generated: output.wav
  ```

### 4. Pretrained Models
Here is a list of pretrained models released by PaddleSpeech that can be used by command and python API:

- Acoustic model
  | Model | Language |
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
  |       fastspeech2_canton     |  canton  |

- Vocoder
  | Model | Language |
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
