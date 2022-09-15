# Paddle Speech Demo

PaddleSpeechDemo 是一个以 PaddleSpeech 的语音交互功能为主体开发的 Demo 展示项目，用于帮助大家更好的上手 PaddleSpeech 以及使用 PaddleSpeech 构建自己的应用。

智能语音交互部分使用 PaddleSpeech，对话以及信息抽取部分使用 PaddleNLP，网页前端展示部分基于 Vue3 进行开发

主要功能：

`main.py` 中包含功能
+ 语音聊天：PaddleSpeech 的语音识别能力+语音合成能力，对话部分基于 PaddleNLP 的闲聊功能
+ 声纹识别：PaddleSpeech 的声纹识别功能展示
+ 语音识别：支持【实时语音识别】，【端到端识别】，【音频文件识别】三种模式
+ 语音合成：支持【流式合成】与【端到端合成】两种方式
+ 语音指令：基于 PaddleSpeech 的语音识别能力与 PaddleNLP 的信息抽取，实现交通费的智能报销

`vc.py` 中包含功能
+ 一句话合成：基于 GE2E 和 ECAPA-TDNN 模型的一句话合成方案，可以模仿输入的音频的音色进行合成任务
+ 小数据微调：基于小数据集的微调方案，内置用10句话标贝中文女声微调示例，你也可以通过一键重置，录制自己的声音，注意在安静环境下录制，效果会更好，你可以在[finetune](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/tts_finetune/tts3)中，使用自己的小数据集，训练音色
+ ENIRE SAT：语言-语音跨模态大模型 ENIRE SAT 可视化展示示例，支持个性化合成，跨语言语音合成（输入音频为中文则合成），语音编辑功能

运行效果：

 ![效果](docs/效果展示.png)

## 安装

### 后端环境安装

```
# 安装环境
cd speech_server
pip install -r requirements.txt

mkdir source
cd source

# 下载 tools 
wget https://paddlespeech.bj.bcebos.com/demos/speech_web/tools.zip
unzip tools.zip

# 下载 wav
wget https://paddlespeech.bj.bcebos.com/demos/speech_web/wav.zip
unzip tools.zip


# 下载 ie 模型，针对地点进行微调
mkdir model
cd model

# 下载IE模型
wget https://bj.bcebos.com/paddlenlp/applications/speech-cmd-analysis/finetune/model_state.pdparams

# 下载 GE2E 相关模型
wget https://bj.bcebos.com/paddlespeech/Parakeet/released_models/ge2e/ge2e_ckpt_0.3.zip
unzip ge2e_ckpt_0.3.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip
unzip pwg_aishell3_ckpt_0.5.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_aishell3_vc1_ckpt_0.5.zip
unzip fastspeech2_nosil_aishell3_vc1_ckpt_0.5.zip

# 下载 SAT 相关模型
# fastspeech2
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_conformer_baker_ckpt_0.5.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_ljspeech_ckpt_0.5.zip
unzip fastspeech2_conformer_baker_ckpt_0.5.zip
unzip fastspeech2_nosil_ljspeech_ckpt_0.5.zip

# aishell3
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_ckpt_0.2.0.zip
unzip hifigan_aishell3_ckpt_0.2.0.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/erniesat_aishell3_ckpt_1.2.0.zip
unzip erniesat_aishell3_ckpt_1.2.0.zip

# vctk
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_vctk_ckpt_0.2.0.zip
unzip unzip hifigan_vctk_ckpt_0.2.0.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/erniesat_vctk_ckpt_1.2.0.zip
unzip erniesat_vctk_ckpt_1.2.0.zip

# aishell3_vctk
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/erniesat_aishell3_vctk_ckpt_1.2.0.zip
unzip erniesat_aishell3_vctk_ckpt_1.2.0.zip

# 下载 finetune 相关模型
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_ckpt_1.1.0.zip
unzip fastspeech2_aishell3_ckpt_1.1.0.zip
```

### 前端环境安装

前端依赖 `node.js` ，需要提前安装，确保 `npm` 可用，`npm` 测试版本 `8.3.1`，建议下载[官网](https://nodejs.org/en/)稳定版的 `node.js`

```
# 进入前端目录
cd web_client

# 安装 `yarn`，已经安装可跳过
npm install -g yarn

# 使用yarn安装前端依赖
yarn install
```

## 启动服务

### 开启后端服务

#### `main.py`
【语音聊天】【声纹识别】【语音识别】【语音合成】【语音指令】功能体验，可直接使用下面的代码
```
cd speech_server
# 默认8010端口
python main.py --port 8010
```

#### `vc.py`

【一句话合成】【小数据微调】【ENIRE SAT】体验都依赖于MFA，体验前先确保 MFA 可用，项目兼容 mfa v1 和 v2 ，source tools中已包含 v1.02版本编译好的工具，如果你是linux系统且mfa可使用，可以将`vc.py`中
```python
sat_model = SAT(mfa_version='v2')
ft_model = FineTune(mfa_version='v2')
```
更改为
```python
sat_model = SAT(mfa_version='v1')
ft_model = FineTune(mfa_version='v1')
```

如果你是其它的系统，可以使用 conda 安装 mfa v2 进行体验，安装请参考 [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html)，确保自己环境中 MFA 可用

```
cd speech_server
# 默认8010端口
python vc.py --port 8010
```

### 开启前端服务

```
cd web_client
yarn dev --port 8011
```

默认配置下，前端中配置的后台地址信息是 localhost，确保后端服务器和打开页面的游览器在同一台机器上，不在一台机器的配置方式见下方的 FAQ：【后端如果部署在其它机器或者别的端口如何修改】
## FAQ 

#### Q: 如何安装node.js

A： node.js的安装可以参考[【菜鸟教程】](https://www.runoob.com/nodejs/nodejs-install-setup.html), 确保 npm 可用

#### Q：后端如果部署在其它机器或者别的端口如何修改

A：后端的配置地址有分散在两个文件中

修改第一个文件 `PaddleSpeechWebClient/vite.config.js`

```
server: {
    host: "0.0.0.0",
    proxy: {
      "/api": {
        target: "http://localhost:8010",  // 这里改成后端所在接口
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  }
```

修改第二个文件 `PaddleSpeechWebClient/src/api/API.js`（ Websocket 代理配置失败，所以需要在这个文件中修改）

```
// websocket （这里改成后端所在的接口）
CHAT_SOCKET_RECORD: 'ws://localhost:8010/ws/asr/offlineStream', // ChatBot websocket 接口
ASR_SOCKET_RECORD: 'ws://localhost:8010/ws/asr/onlineStream',  // Stream ASR 接口
TTS_SOCKET_RECORD: 'ws://localhost:8010/ws/tts/online', // Stream TTS 接口
```

#### Q：后端以IP地址的形式，前端无法录音

A：这里主要是游览器安全策略的限制，需要配置游览器后重启。游览器修改配置可参考[使用js-audio-recorder报浏览器不支持getUserMedia](https://blog.csdn.net/YRY_LIKE_YOU/article/details/113745273)

chrome设置地址: chrome://flags/#unsafely-treat-insecure-origin-as-secure

## 参考资料

vue实现录音参考资料：https://blog.csdn.net/qq_41619796/article/details/107865602#t1

前端流式播放音频参考仓库：

https://github.com/AnthumChris/fetch-stream-audio

https://bm.enthuses.me/buffered.php?bref=6677
