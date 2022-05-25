(简体中文|[English](./README.md))
<p align="center">
  <img src="./docs/images/PaddleSpeech_logo.png" />
</p>


<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSpeech/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleSpeech?color=ffa"></a>
    <a href="support os"><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSpeech/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleSpeech?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSpeech/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleSpeech?color=3af"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSpeech/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleSpeech?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleSpeech/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleSpeech?color=ccf"></a>
    <a href="=https://pypi.org/project/paddlespeech/"><img src="https://img.shields.io/pypi/dm/PaddleSpeech"></a>
    <a href="=https://pypi.org/project/paddlespeech/"><img src="https://static.pepy.tech/badge/paddlespeech"></a>
    <a href="https://huggingface.co/spaces"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"></a>
</p>
<div align="center">  
<h4>
    <a href="#快速开始"> 快速开始 </a>
  | <a href="#快速使用服务"> 快速使用服务 </a>
  | <a href="#快速使用流式服务"> 快速使用流式服务 </a>
  | <a href="#教程文档"> 教程文档 </a>
  | <a href="#模型列表"> 模型列表 </a>
  | <a href="https://aistudio.baidu.com/aistudio/education/group/info/25130"> AIStudio 课程 </a>
</h4>
</div>


------------------------------------------------------------------------------------

**PaddleSpeech** 是基于飞桨 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) 的语音方向的开源模型库，用于语音和音频中的各种关键任务的开发，包含大量基于深度学习前沿和有影响力的模型，一些典型的应用示例如下：
##### 语音识别

<div align = "center">
<table style="width:100%">
  <thead>
    <tr>
      <th> 输入音频  </th>
      <th width="550"> 识别结果 </th>
    </tr>
  </thead>
  <tbody>
   <tr>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200 style="max-width: 100%;"></a><br>
      </td>
      <td >I knocked at the door on the ancient side of the building.</td>
    </tr>
    <tr>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
      <td>我认为跑步最重要的就是给我带来了身体健康。</td>
    </tr>
  </tbody>
</table>

</div>

##### 语音翻译 (英译中)

<div align = "center">
<table style="width:100%">
  <thead>
    <tr>
      <th> 输入音频 </th>
      <th width="550"> 翻译结果 </th>
    </tr>
  </thead>
  <tbody>
   <tr>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200 style="max-width: 100%;"></a><br>
      </td>
      <td >我 在 这栋 建筑 的 古老 门上 敲门。</td>
    </tr>
  </tbody>
</table>

</div>

##### 语音合成
<div align = "center">
<table style="width:100%">
  <thead>
    <tr>
      <th width="550">输入文本</th>
      <th>合成音频</th>
    </tr>
  </thead>
  <tbody>
   <tr>
      <td >Life was like a box of chocolates, you never know what you're gonna get.</td>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_1.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
    <tr>
      <td >早上好，今天是2020/10/29，最低温度是-3°C。</td>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/parakeet_espnet_fs2_pwg_demo/tn_g2p/parakeet/001.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
    <tr>
      <td >季姬寂，集鸡，鸡即棘鸡。棘鸡饥叽，季姬及箕稷济鸡。鸡既济，跻姬笈，季姬忌，急咭鸡，鸡急，继圾几，季姬急，即籍箕击鸡，箕疾击几伎，伎即齑，鸡叽集几基，季姬急极屐击鸡，鸡既殛，季姬激，即记《季姬击鸡记》。</td>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/jijiji.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
  </tbody>
</table>

</div>

更多合成音频，可以参考 [PaddleSpeech 语音合成音频示例](https://paddlespeech.readthedocs.io/en/latest/tts/demo.html)。

##### 标点恢复
<div align = "center">
<table style="width:100%">
  <thead>
    <tr>
      <th width="390"> 输入文本 </th>
      <th width="390"> 输出文本 </th>
    </tr>
  </thead>
  <tbody>
   <tr>
      <td>今天的天气真不错啊你下午有空吗我想约你一起去吃饭</td>
      <td>今天的天气真不错啊！你下午有空吗？我想约你一起去吃饭。</td>
    </tr>
  </tbody>
</table>

</div>


### 特性

本项目采用了易用、高效、灵活以及可扩展的实现，旨在为工业应用、学术研究提供更好的支持，实现的功能包含训练、推断以及测试模块，以及部署过程，主要包括
- 📦 **易用性**: 安装门槛低，可使用 [CLI](#quick-start) 快速开始。
- 🏆 **对标 SoTA**: 提供了高速、轻量级模型，且借鉴了最前沿的技术。
- 🏆 **流式ASR和TTS系统**：工业级的端到端流式识别、流式合成系统。
- 💯 **基于规则的中文前端**: 我们的前端包含文本正则化和字音转换（G2P）。此外，我们使用自定义语言规则来适应中文语境。
- **多种工业界以及学术界主流功能支持**:
  - 🛎️ 典型音频任务: 本工具包提供了音频任务如音频分类、语音翻译、自动语音识别、文本转语音、语音合成、声纹识别、KWS等任务的实现。
  - 🔬 主流模型及数据集: 本工具包实现了参与整条语音任务流水线的各个模块，并且采用了主流数据集如 LibriSpeech、LJSpeech、AIShell、CSMSC，详情请见 [模型列表](#model-list)。
  - 🧩 级联模型应用: 作为传统语音任务的扩展，我们结合了自然语言处理、计算机视觉等任务，实现更接近实际需求的产业级应用。


### 近期更新
- 👑 2022.05.13: PaddleSpeech 发布 [PP-ASR](./docs/source/asr/PPASR_cn.md) 流式语音识别系统、[PP-TTS](./docs/source/tts/PPTTS_cn.md) 流式语音合成系统、[PP-VPR](docs/source/vpr/PPVPR_cn.md) 全链路声纹识别系统
- 👏🏻 2022.05.06: PaddleSpeech Streaming Server 上线! 覆盖了语音识别（标点恢复、时间戳），和语音合成。
- 👏🏻 2022.05.06: PaddleSpeech Server 上线! 覆盖了声音分类、语音识别、语音合成、声纹识别，标点恢复。
- 👏🏻 2022.03.28: PaddleSpeech CLI 覆盖声音分类、语音识别、语音翻译（英译中）、语音合成，声纹验证。
- 🤗 2021.12.14: PaddleSpeech [ASR](https://huggingface.co/spaces/KPatrick/PaddleSpeechASR) and [TTS](https://huggingface.co/spaces/KPatrick/PaddleSpeechTTS) Demos on Hugging Face Spaces are available!


 ### 🔥 加入技术交流群获取入群福利

 - 3 日直播课链接: 深度解读 PP-TTS、PP-ASR、PP-VPR 三项核心语音系统关键技术
 - 20G 学习大礼包：视频课程、前沿论文与学习资料
  
微信扫描二维码关注公众号，点击“马上报名”填写问卷加入官方交流群，获得更高效的问题答疑，与各行各业开发者充分交流，期待您的加入。

<div align="center">
<img src="https://user-images.githubusercontent.com/23690325/169763015-cbd8e28d-602c-4723-810d-dbc6da49441e.jpg"  width = "200"  />
</div>

## 安装

我们强烈建议用户在 **Linux** 环境下，*3.7* 以上版本的 *python* 上安装 PaddleSpeech。
目前为止，**Linux** 支持声音分类、语音识别、语音合成和语音翻译四种功能，**Mac OSX、 Windows** 下暂不支持语音翻译功能。 想了解具体安装细节，可以参考[安装文档](./docs/source/install_cn.md)。

<a name="快速开始"></a>
## 快速开始

安装完成后，开发者可以通过命令行快速开始，改变 `--input` 可以尝试用自己的音频或文本测试。

**声音分类**     
```shell
paddlespeech cls --input input.wav
```
**声纹识别**
```shell
paddlespeech vector --task spk --input input_16k.wav
```
**语音识别**
```shell
paddlespeech asr --lang zh --input input_16k.wav
```
**语音翻译** (English to Chinese)
```shell
paddlespeech st --input input_16k.wav
```
**语音合成** 
```shell
paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav
```
- 语音合成的 web demo 已经集成进了 [Huggingface Spaces](https://huggingface.co/spaces). 请参考: [TTS Demo](https://huggingface.co/spaces/akhaliq/paddlespeech)

**文本后处理** 
 - 标点恢复
   ```bash
   paddlespeech text --task punc --input 今天的天气真不错啊你下午有空吗我想约你一起去吃饭
   ```

**批处理**
```
echo -e "1 欢迎光临。\n2 谢谢惠顾。" | paddlespeech tts
```

**Shell管道**
ASR + Punc:
```
paddlespeech asr --input ./zh.wav | paddlespeech text --task punc
```

更多命令行命令请参考 [demos](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos)
> Note: 如果需要训练或者微调，请查看[语音识别](./docs/source/asr/quick_start.md)， [语音合成](./docs/source/tts/quick_start.md)。

<a name="快速使用服务"></a>
## 快速使用服务
安装完成后，开发者可以通过命令行快速使用服务。

**启动服务**     
```shell
paddlespeech_server start --config_file ./paddlespeech/server/conf/application.yaml
```

**访问语音识别服务**     
```shell
paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input input_16k.wav
```

**访问语音合成服务**     
```shell
paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
```

**访问音频分类服务**     
```shell
paddlespeech_client cls --server_ip 127.0.0.1 --port 8090 --input input.wav
```

更多服务相关的命令行使用信息，请参考 [demos](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/speech_server)

<a name="快速使用流式服务"></a>
## 快速使用流式服务

开发者可以尝试 [流式 ASR](./demos/streaming_asr_server/README.md) 和 [流式 TTS](./demos/streaming_tts_server/README.md) 服务.

**启动流式 ASR 服务**

```
paddlespeech_server start --config_file ./demos/streaming_asr_server/conf/application.yaml
```

**访问流式 ASR 服务**     

```
paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8090 --input input_16k.wav
```

**启动流式 TTS 服务**

```
paddlespeech_server start --config_file ./demos/streaming_tts_server/conf/tts_online_application.yaml
```

**访问流式 TTS 服务**     

```
paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8092 --protocol http --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
```

更多信息参看： [流式 ASR](./demos/streaming_asr_server/README.md) 和 [流式 TTS](./demos/streaming_tts_server/README.md) 

<a name="模型列表"></a>
## 模型列表
PaddleSpeech 支持很多主流的模型，并提供了预训练模型，详情请见[模型列表](./docs/source/released_model.md)。

<a name="语音识别模型"></a>

PaddleSpeech 的 **语音转文本** 包含语音识别声学模型、语音识别语言模型和语音翻译, 详情如下：

<table style="width:100%">
  <thead>
    <tr>
      <th>语音转文本模块类型</th>
      <th>数据集</th>
      <th>模型类型</th>
      <th>脚本</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">语音识别</td>
      <td rowspan="2" >Aishell</td>
      <td >DeepSpeech2 RNN + Conv based Models</td>
      <td>
      <a href = "./examples/aishell/asr0">deepspeech2-aishell</a>
      </td>
    </tr>
    <tr>
      <td>Transformer based Attention Models </td>
      <td>
      <a href = "./examples/aishell/asr1">u2.transformer.conformer-aishell</a>
      </td>
    </tr>
      <tr>
      <td> Librispeech</td>
      <td>Transformer based Attention Models </td>
      <td>
      <a href = "./examples/librispeech/asr0">deepspeech2-librispeech</a> / <a href = "./examples/librispeech/asr1">transformer.conformer.u2-librispeech</a>  / <a href = "./examples/librispeech/asr2">transformer.conformer.u2-kaldi-librispeech</a>
      </td>
      </td>
    </tr>
    <tr>
      <td>TIMIT</td>
      <td>Unified Streaming & Non-streaming Two-pass</td>
      <td>
    <a href = "./examples/timit/asr1"> u2-timit</a>
      </td>
    </tr>
  <tr>
  <td>对齐</td>
  <td>THCHS30</td>
  <td>MFA</td>
  <td>
  <a href = ".examples/thchs30/align0">mfa-thchs30</a>
  </td>
  </tr>
   <tr>
      <td rowspan="1">语言模型</td>
      <td colspan = "2">Ngram 语言模型</td>
      <td>
      <a href = "./examples/other/ngram_lm">kenlm</a>
      </td>
    </tr>
    <tr>
      <td rowspan="2">语音翻译（英译中）</td> 
      <td rowspan="2">TED En-Zh</td>
      <td>Transformer + ASR MTL</td>
      <td>
      <a href = "./examples/ted_en_zh/st0">transformer-ted</a>
      </td>
  </tr>
  <tr>
      <td>FAT + Transformer + ASR MTL</td>
      <td>
      <a href = "./examples/ted_en_zh/st1">fat-st-ted</a>
      </td>
  </tr>
  </tbody>
</table>

<a name="语音合成模型"></a>

PaddleSpeech 的 **语音合成** 主要包含三个模块：文本前端、声学模型和声码器。声学模型和声码器模型如下：

<table>
  <thead>
    <tr>
      <th> 语音合成模块类型 </th>
      <th> 模型类型 </th>
      <th> 数据集  </th>
      <th> 脚本  </th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td> 文本前端</td>
    <td colspan="2"> &emsp; </td>
    <td>
    <a href = "./examples/other/tn">tn</a> / <a href = "./examples/other/g2p">g2p</a>
    </td>
    </tr>
    <tr>
      <td rowspan="4">声学模型</td>
      <td>Tacotron2</td>
      <td>LJSpeech / CSMSC</td>
      <td>
      <a href = "./examples/ljspeech/tts0">tacotron2-ljspeech</a> / <a href = "./examples/csmsc/tts0">tacotron2-csmsc</a>
      </td>
    </tr>
    <tr>
      <td>Transformer TTS</td>
      <td>LJSpeech</td>
      <td>
      <a href = "./examples/ljspeech/tts1">transformer-ljspeech</a>
      </td>
    </tr>
    <tr>
      <td>SpeedySpeech</td>
      <td>CSMSC</td>
      <td >
      <a href = "./examples/csmsc/tts2">speedyspeech-csmsc</a>
      </td>
    </tr>
    <tr>
      <td>FastSpeech2</td>
      <td>LJSpeech / VCTK / CSMSC / AISHELL-3</td>
      <td>
      <a href = "./examples/ljspeech/tts3">fastspeech2-ljspeech</a> / <a href = "./examples/vctk/tts3">fastspeech2-vctk</a> / <a href = "./examples/csmsc/tts3">fastspeech2-csmsc</a> / <a href = "./examples/aishell3/tts3">fastspeech2-aishell3</a>
      </td>
    </tr>
   <tr>
      <td rowspan="6">声码器</td>
      <td >WaveFlow</td>
      <td >LJSpeech</td>
      <td>
      <a href = "./examples/ljspeech/voc0">waveflow-ljspeech</a>
      </td>
    </tr>
    <tr>
      <td >Parallel WaveGAN</td>
      <td >LJSpeech / VCTK / CSMSC / AISHELL-3</td>
      <td>
      <a href = "./examples/ljspeech/voc1">PWGAN-ljspeech</a> / <a href = "./examples/vctk/voc1">PWGAN-vctk</a> / <a href = "./examples/csmsc/voc1">PWGAN-csmsc</a> /  <a href = "./examples/aishell3/voc1">PWGAN-aishell3</a>
      </td>
    </tr>
    <tr>
      <td >Multi Band MelGAN</td>
      <td >CSMSC</td>
      <td>
      <a href = "./examples/csmsc/voc3">Multi Band MelGAN-csmsc</a> 
      </td>
    </tr>
    <tr>
      <td >Style MelGAN</td>
      <td >CSMSC</td>
      <td>
      <a href = "./examples/csmsc/voc4">Style MelGAN-csmsc</a> 
      </td>
    </tr>
    <tr>
      <td >HiFiGAN</td>
      <td >LJSpeech / VCTK / CSMSC / AISHELL-3</td>
      <td>
      <a href = "./examples/ljspeech/voc5">HiFiGAN-ljspeech</a> / <a href = "./examples/vctk/voc5">HiFiGAN-vctk</a> / <a href = "./examples/csmsc/voc5">HiFiGAN-csmsc</a> / <a href = "./examples/aishell3/voc5">HiFiGAN-aishell3</a>
      </td>
    </tr>
    <tr>
      <td >WaveRNN</td>
      <td >CSMSC</td>
      <td>
      <a href = "./examples/csmsc/voc6">WaveRNN-csmsc</a>
      </td>
    </tr>
    <tr>
      <td rowspan="3">声音克隆</td>
      <td>GE2E</td>
      <td >Librispeech, etc.</td>
      <td>
      <a href = "./examples/other/ge2e">ge2e</a>
      </td>
    </tr>
    <tr>
      <td>GE2E + Tacotron2</td>
      <td>AISHELL-3</td>
      <td>
      <a href = "./examples/aishell3/vc0">ge2e-tacotron2-aishell3</a>
      </td>
    </tr>
    <tr>
      <td>GE2E + FastSpeech2</td>
      <td>AISHELL-3</td>
      <td>
      <a href = "./examples/aishell3/vc1">ge2e-fastspeech2-aishell3</a>
      </td>
    </tr>
  </tbody>
</table>

<a name="声音分类模型"></a>
**声音分类**

<table style="width:100%">
  <thead>
    <tr>
      <th> 任务 </th>
      <th> 数据集 </th>
      <th> 模型类型 </th>
      <th> 脚本</th>
    </tr>
  </thead>
  <tbody>
  <tr>
      <td>声音分类</td>
      <td>ESC-50</td>
      <td>PANN</td>
      <td>
      <a href = "./examples/esc50/cls0">pann-esc50</a>
      </td>
    </tr>
  </tbody>
</table>


<a name="声纹识别模型"></a>

**声纹识别**

<table style="width:100%">
  <thead>
    <tr>
      <th> 任务 </th>
      <th> 数据集 </th>
      <th> 模型类型 </th>
      <th> 脚本 </th>
    </tr>
  </thead>
  <tbody>
  <tr>
      <td>Speaker Verification</td>
      <td>VoxCeleb12</td>
      <td>ECAPA-TDNN</td>
      <td>
      <a href = "./examples/voxceleb/sv0">ecapa-tdnn-voxceleb12</a>
      </td>
    </tr>
  </tbody>
</table>

<a name="标点恢复模型"></a>

**标点恢复**

<table style="width:100%">
  <thead>
    <tr>
      <th> 任务 </th>
      <th> 数据集 </th>
      <th> 模型类型 </th>
      <th> 脚本 </th>
    </tr>
  </thead>
  <tbody>
  <tr>
      <td>标点恢复</td>
      <td>IWLST2012_zh</td>
      <td>Ernie Linear</td>
      <td>
      <a href = "./examples/iwslt2012/punc0">iwslt2012-punc0</a>
      </td>
    </tr>
  </tbody>
</table>

<a name="教程文档"></a>
## 教程文档

对于 PaddleSpeech 的所关注的任务，以下指南有助于帮助开发者快速入门，了解语音相关核心思想。

- [下载安装](./docs/source/install_cn.md)
- [快速开始](#快速开始)
- Notebook基础教程
  - [声音分类](./docs/tutorial/cls/cls_tutorial.ipynb)
  - [语音识别](./docs/tutorial/asr/tutorial_transformer.ipynb)
  - [语音翻译](./docs/tutorial/st/st_tutorial.ipynb)
  - [声音合成](./docs/tutorial/tts/tts_tutorial.ipynb)
  - [示例Demo](./demos/README.md)
- 进阶文档  
  - [语音识别自定义训练](./docs/source/asr/quick_start.md)
    - [简介](./docs/source/asr/models_introduction.md)
    - [数据准备](./docs/source/asr/data_preparation.md)
    - [Ngram 语言模型](./docs/source/asr/ngram_lm.md)
  - [语音合成自定义训练](./docs/source/tts/quick_start.md)
    - [简介](./docs/source/tts/models_introduction.md)
    - [进阶用法](./docs/source/tts/advanced_usage.md)
    - [中文文本前端](./docs/source/tts/zh_text_frontend.md)
    - [测试语音样本](https://paddlespeech.readthedocs.io/en/latest/tts/demo.html)
  - 声纹识别
    - [声纹识别](./demos/speaker_verification/README_cn.md)
    - [音频检索](./demos/audio_searching/README_cn.md)
  - [声音分类](./demos/audio_tagging/README_cn.md)
  - [语音翻译](./demos/speech_translation/README_cn.md)
  - [服务化部署](./demos/speech_server/README_cn.md)
- [模型列表](#模型列表)
  - [语音识别](#语音识别模型)
  - [语音合成](#语音合成模型)
  - [声音分类](#声音分类模型)
  - [声纹识别](#声纹识别模型)
  - [标点恢复](#标点恢复模型)
- [技术交流群](#技术交流群)
- [欢迎贡献](#欢迎贡献)
- [License](#License)


语音合成模块最初被称为 [Parakeet](https://github.com/PaddlePaddle/Parakeet)，现在与此仓库合并。如果您对该任务的学术研究感兴趣，请参阅 [TTS 研究概述](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/docs/source/tts#overview)。此外，[模型介绍](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/tts/models_introduction.md) 是了解语音合成流程的一个很好的指南。

## ⭐ 应用案例
- **[PaddleBoBo](https://github.com/JiehangXie/PaddleBoBo): 使用 PaddleSpeech 的语音合成模块生成虚拟人的声音。**
  
<div align="center"><a href="https://www.bilibili.com/video/BV1cL411V71o?share_source=copy_web"><img src="https://ai-studio-static-online.cdn.bcebos.com/06fd746ab32042f398fb6f33f873e6869e846fe63c214596ae37860fe8103720" / width="500px"></a></div>

- [PaddleSpeech 示例视频](https://paddlespeech.readthedocs.io/en/latest/demo_video.html)


- **[VTuberTalk](https://github.com/jerryuhoo/VTuberTalk): 使用 PaddleSpeech 的语音合成和语音识别从视频中克隆人声。**

<div align="center">
<img src="https://raw.githubusercontent.com/jerryuhoo/VTuberTalk/main/gui/gui.png"  width = "500px"  />
</div>


## 引用

要引用 PaddleSpeech 进行研究，请使用以下格式进行引用。
```text
@inproceedings{zhang2022paddlespeech,
    title = {PaddleSpeech: An Easy-to-Use All-in-One Speech Toolkit},
    author = {Hui Zhang, Tian Yuan, Junkun Chen, Xintong Li, Renjie Zheng, Yuxin Huang, Xiaojie Chen, Enlei Gong, Zeyu Chen, Xiaoguang Hu, dianhai yu, Yanjun Ma, Liang Huang},
    booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Demonstrations},
    year = {2022},
    publisher = {Association for Computational Linguistics},
}

@inproceedings{zheng2021fused,
  title={Fused acoustic and text encoding for multimodal bilingual pretraining and speech translation},
  author={Zheng, Renjie and Chen, Junkun and Ma, Mingbo and Huang, Liang},
  booktitle={International Conference on Machine Learning},
  pages={12736--12746},
  year={2021},
  organization={PMLR}
}
```

<a name="欢迎贡献"></a>
## 参与 PaddleSpeech 的开发

热烈欢迎您在 [Discussions](https://github.com/PaddlePaddle/PaddleSpeech/discussions) 中提交问题，并在 [Issues](https://github.com/PaddlePaddle/PaddleSpeech/issues) 中指出发现的 bug。此外，我们非常希望您参与到 PaddleSpeech 的开发中！

### 贡献者
<p align="center">
<a href="https://github.com/zh794390558"><img src="https://avatars.githubusercontent.com/u/3038472?v=4" width=75 height=75></a>
<a href="https://github.com/Jackwaterveg"><img src="https://avatars.githubusercontent.com/u/87408988?v=4" width=75 height=75></a>
<a href="https://github.com/yt605155624"><img src="https://avatars.githubusercontent.com/u/24568452?v=4" width=75 height=75></a>
<a href="https://github.com/kuke"><img src="https://avatars.githubusercontent.com/u/3064195?v=4" width=75 height=75></a>
<a href="https://github.com/xinghai-sun"><img src="https://avatars.githubusercontent.com/u/7038341?v=4" width=75 height=75></a>
<a href="https://github.com/pkuyym"><img src="https://avatars.githubusercontent.com/u/5782283?v=4" width=75 height=75></a>
<a href="https://github.com/KPatr1ck"><img src="https://avatars.githubusercontent.com/u/22954146?v=4" width=75 height=75></a>
<a href="https://github.com/LittleChenCc"><img src="https://avatars.githubusercontent.com/u/10339970?v=4" width=75 height=75></a>
<a href="https://github.com/745165806"><img src="https://avatars.githubusercontent.com/u/20623194?v=4" width=75 height=75></a>
<a href="https://github.com/Mingxue-Xu"><img src="https://avatars.githubusercontent.com/u/92848346?v=4" width=75 height=75></a>
<a href="https://github.com/chrisxu2016"><img src="https://avatars.githubusercontent.com/u/18379485?v=4" width=75 height=75></a>
<a href="https://github.com/lfchener"><img src="https://avatars.githubusercontent.com/u/6771821?v=4" width=75 height=75></a>
<a href="https://github.com/luotao1"><img src="https://avatars.githubusercontent.com/u/6836917?v=4" width=75 height=75></a>
<a href="https://github.com/wanghaoshuang"><img src="https://avatars.githubusercontent.com/u/7534971?v=4" width=75 height=75></a>
<a href="https://github.com/gongel"><img src="https://avatars.githubusercontent.com/u/24390500?v=4" width=75 height=75></a>
<a href="https://github.com/mmglove"><img src="https://avatars.githubusercontent.com/u/38800877?v=4" width=75 height=75></a>
<a href="https://github.com/iclementine"><img src="https://avatars.githubusercontent.com/u/16222986?v=4" width=75 height=75></a>
<a href="https://github.com/ZeyuChen"><img src="https://avatars.githubusercontent.com/u/1371212?v=4" width=75 height=75></a>
<a href="https://github.com/AK391"><img src="https://avatars.githubusercontent.com/u/81195143?v=4" width=75 height=75></a>
<a href="https://github.com/qingqing01"><img src="https://avatars.githubusercontent.com/u/7845005?v=4" width=75 height=75></a>
<a href="https://github.com/ericxk"><img src="https://avatars.githubusercontent.com/u/4719594?v=4" width=75 height=75></a>
<a href="https://github.com/kvinwang"><img src="https://avatars.githubusercontent.com/u/6442159?v=4" width=75 height=75></a>
<a href="https://github.com/jiqiren11"><img src="https://avatars.githubusercontent.com/u/82639260?v=4" width=75 height=75></a>
<a href="https://github.com/AshishKarel"><img src="https://avatars.githubusercontent.com/u/58069375?v=4" width=75 height=75></a>
<a href="https://github.com/chesterkuo"><img src="https://avatars.githubusercontent.com/u/6285069?v=4" width=75 height=75></a>
<a href="https://github.com/tensor-tang"><img src="https://avatars.githubusercontent.com/u/21351065?v=4" width=75 height=75></a>
<a href="https://github.com/hysunflower"><img src="https://avatars.githubusercontent.com/u/52739577?v=4" width=75 height=75></a>  
<a href="https://github.com/wwhu"><img src="https://avatars.githubusercontent.com/u/6081200?v=4" width=75 height=75></a>
<a href="https://github.com/lispc"><img src="https://avatars.githubusercontent.com/u/2833376?v=4" width=75 height=75></a>
<a href="https://github.com/jerryuhoo"><img src="https://avatars.githubusercontent.com/u/24245709?v=4" width=75 height=75></a>
<a href="https://github.com/harisankarh"><img src="https://avatars.githubusercontent.com/u/1307053?v=4" width=75 height=75></a>
<a href="https://github.com/Jackiexiao"><img src="https://avatars.githubusercontent.com/u/18050469?v=4" width=75 height=75></a>
<a href="https://github.com/limpidezza"><img src="https://avatars.githubusercontent.com/u/71760778?v=4" width=75 height=75></a>
</p>

## 致谢

- 非常感谢 [yeyupiaoling](https://github.com/yeyupiaoling)/[PPASR](https://github.com/yeyupiaoling/PPASR)/[PaddlePaddle-DeepSpeech](https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech)/[VoiceprintRecognition-PaddlePaddle](https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)/[AudioClassification-PaddlePaddle](https://github.com/yeyupiaoling/AudioClassification-PaddlePaddle) 多年来的关注和建议，以及在诸多问题上的帮助。
- 非常感谢 [mymagicpower](https://github.com/mymagicpower) 采用PaddleSpeech 对 ASR 的[短语音](https://github.com/mymagicpower/AIAS/tree/main/3_audio_sdks/asr_sdk)及[长语音](https://github.com/mymagicpower/AIAS/tree/main/3_audio_sdks/asr_long_audio_sdk)进行 Java 实现。
- 非常感谢 [JiehangXie](https://github.com/JiehangXie)/[PaddleBoBo](https://github.com/JiehangXie/PaddleBoBo) 采用 PaddleSpeech 语音合成功能实现 Virtual Uploader(VUP)/Virtual YouTuber(VTuber) 虚拟主播。
- 非常感谢 [745165806](https://github.com/745165806)/[PaddleSpeechTask](https://github.com/745165806/PaddleSpeechTask) 贡献标点重建相关模型。
- 非常感谢 [kslz](https://github.com/kslz) 补充中文文档。
- 非常感谢 [awmmmm](https://github.com/awmmmm) 提供 fastspeech2 aishell3 conformer 预训练模型。
- 非常感谢 [phecda-xu](https://github.com/phecda-xu)/[PaddleDubbing](https://github.com/phecda-xu/PaddleDubbing) 基于 PaddleSpeech 的 TTS 模型搭建带 GUI 操作界面的配音工具。
- 非常感谢 [jerryuhoo](https://github.com/jerryuhoo)/[VTuberTalk](https://github.com/jerryuhoo/VTuberTalk) 基于 PaddleSpeech 的 TTS GUI 界面和基于 ASR 制作数据集的相关代码。

  

此外，PaddleSpeech 依赖于许多开源存储库。有关更多信息，请参阅 [references](./docs/source/reference.md)。

## License

PaddleSpeech 在 [Apache-2.0 许可](./LICENSE) 下提供。
