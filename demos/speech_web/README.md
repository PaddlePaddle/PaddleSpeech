# Paddle Speech Demo

## 简介
Paddle Speech Demo 是一个以 PaddleSpeech 的语音交互功能为主体开发的 Demo 展示项目，用于帮助大家更好的上手 PaddleSpeech 以及使用 PaddleSpeech 构建自己的应用。

智能语音交互部分使用 PaddleSpeech，对话以及信息抽取部分使用 PaddleNLP，网页前端展示部分基于 Vue3 进行开发。

主要功能：

`main.py` 中包含功能
+ 语音聊天：PaddleSpeech 的语音识别能力+语音合成能力，对话部分基于 PaddleNLP 的闲聊功能
+ 声纹识别：PaddleSpeech 的声纹识别功能展示
+ 语音识别：支持【实时语音识别】，【端到端识别】，【音频文件识别】三种模式
+ 语音合成：支持【流式合成】与【端到端合成】两种方式
+ 语音指令：基于 PaddleSpeech 的语音识别能力与 PaddleNLP 的信息抽取，实现交通费的智能报销

`vc.py` 中包含功能
+ 一句话合成：基于 GE2E 和 ECAPA-TDNN 模型的一句话合成方案，可以模仿输入的音频的音色进行合成任务
  + GE2E 音色克隆方案可以参考： [【FastSpeech2 + AISHELL-3 Voice Cloning】](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/vc1)
  + ECAPA-TDNN 音色克隆方案可以参考: [【FastSpeech2 + AISHELL-3 Voice Cloning (ECAPA-TDNN)】](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/vc2)

+ 小数据微调：基于小数据集的微调方案，内置用12句话标贝中文女声微调示例，你也可以通过一键重置，录制自己的声音，注意在安静环境下录制，效果会更好。你可以在 [【Finetune your own AM based on FastSpeech2 with AISHELL-3】](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/tts_finetune/tts3)中尝试使用自己的数据集进行微调。

+ ERNIE-SAT：语言-语音跨模态大模型 ERNIE-SAT 可视化展示示例，支持个性化合成，跨语言语音合成（音频为中文则输入英文文本进行合成），语音编辑（修改音频文字中间的结果）功能。 ERNIE-SAT 更多实现细节，可以参考：
  + [【ERNIE-SAT with AISHELL-3 dataset】](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/ernie_sat)
  + [【ERNIE-SAT with AISHELL3 and VCTK datasets】](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3_vctk/ernie_sat)
  + [【ERNIE-SAT with VCTK dataset】](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/vctk/ernie_sat)

运行效果：

 ![效果](https://user-images.githubusercontent.com/30135920/196076507-7eb33d39-2345-4268-aee7-6270b9ac8b98.png)



## 基础环境安装

### 后端环境安装
```bash 
# 需要先安装 PaddleSpeech
cd speech_server
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
cd ../
```

### 前端环境安装
前端依赖 `node.js` ，需要提前安装，确保 `npm` 可用，`npm` 测试版本 `8.3.1`，建议下载[官网](https://nodejs.org/en/)稳定版的 `node.js`

如果因为网络问题，无法下载依赖库，可以参考 FAQ 部分，`npm / yarn 下载速度慢问题`

```bash
# 进入前端目录
cd web_client
# 安装 `yarn`，已经安装可跳过
npm install -g yarn
# 使用yarn安装前端依赖
yarn install
cd ../
```


## 启动服务
【注意】目前只支持 `main.py` 和 `vc.py` 两者中选择开启一个后端服务。

### 启动 `main.py` 后端服务

#### 下载相关模型

只需手动下载语音指令所需模型即可，其他模型会自动下载。

```bash
cd speech_server
mkdir -p source/model
cd source/model
# 下载IE模型
wget https://bj.bcebos.com/paddlenlp/applications/speech-cmd-analysis/finetune/model_state.pdparams
cd ../../../

```
#### 启动后端服务

```
cd speech_server
# 默认8010端口
python main.py --port 8010
```


### 启动 `vc.py` 后端服务

参照下面的步骤自行配置项目所需环境。

Aistudio 在线体验小样本合成后端功能：[【PaddleSpeech进阶】PaddleSpeech小样本合成方案体验](https://aistudio.baidu.com/aistudio/projectdetail/4573549?sUid=2470186&shared=1&ts=1664174385948)

#### 下载相关模型和音频

```bash
cd speech_server

# 已创建则跳过
mkdir -p source/model
cd source
# 下载 & 解压 wav （包含VC测试音频）
wget https://paddlespeech.bj.bcebos.com/demos/speech_web/wav_vc.zip
unzip wav_vc.zip

cd model
# 下载 GE2E 相关模型
wget https://bj.bcebos.com/paddlespeech/Parakeet/released_models/ge2e/ge2e_ckpt_0.3.zip
unzip ge2e_ckpt_0.3.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip
unzip pwg_aishell3_ckpt_0.5.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_aishell3_vc1_ckpt_0.5.zip
unzip fastspeech2_nosil_aishell3_vc1_ckpt_0.5.zip

# 下载 ECAPA-TDNN 相关模型
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_ckpt_vc2_1.2.0.zip
unzip fastspeech2_aishell3_ckpt_vc2_1.2.0.zip

# 下载 ERNIE-SAT 相关模型
# aishell3 ERNIE-SAT
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/erniesat_aishell3_ckpt_1.2.0.zip
unzip erniesat_aishell3_ckpt_1.2.0.zip

# vctk ERNIE-SAT
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/erniesat_vctk_ckpt_1.2.0.zip
unzip erniesat_vctk_ckpt_1.2.0.zip

# aishell3_vctk ERNIE-SAT
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/erniesat_aishell3_vctk_ckpt_1.2.0.zip
unzip erniesat_aishell3_vctk_ckpt_1.2.0.zip

# 下载 finetune 相关模型
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_ckpt_1.1.0.zip
unzip fastspeech2_aishell3_ckpt_1.1.0.zip

# 下载声码器
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_ckpt_0.2.0.zip
unzip hifigan_aishell3_ckpt_0.2.0.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_vctk_ckpt_0.2.0.zip
unzip hifigan_vctk_ckpt_0.2.0.zip

cd ../../../
```

#### ERNIE-SAT 环境配置

ERNIE-SAT 体验依赖于 [examples/aishell3_vctk/ernie_sat](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3_vctk/ernie_sat) 的环境。参考 `examples/aishell3_vctk/ernie_sat` 下的 `README.md`， 确保 `examples/aishell3_vctk/ernie_sat` 下 `run.sh` 相关示例代码有效。
 
运行好 `examples/aishell3_vctk/ernie_sat` 后，回到当前目录，创建环境：
```bash
cd speech_server
ln -snf ../../../examples/aishell3_vctk/ernie_sat/download .
ln -snf ../../../examples/aishell3_vctk/ernie_sat/tools .
cd ../
```

#### finetune 环境配置

`finetune` 需要解压 `tools/aligner` 中的 `aishell3_model.zip`，finetune 过程需要使用到 `tools/aligner/aishell3_model/meta.yaml` 文件。

```bash
cd speech_server/tools/aligner
unzip aishell3_model.zip
cd -
```

#### 启动后端服务

```
cd speech_server
# 默认8010端口
python vc.py --port 8010
```

### 启动前端服务

```
cd web_client
yarn dev --port 8011
```

默认配置下，前端配置的后台地址信息是 `localhost`，确保后端服务器和打开页面的游览器在同一台机器上，不在一台机器的配置方式见下方的 FAQ：【后端如果部署在其它机器或者别的端口如何修改】

#### 关于前端的一些说明

为了方便后期的维护，这里并没有给出打包好的 HTML 文件，而是 Vue3 的项目，使用 `yarn dev --port 8011` 的方式启动测试，方便大家debug，相当于是启动了一个前端服务器。

比如我们在本机启动的这个前端服务（运行 `yarn dev --port 8011` ），我们就可以通过在游览器中通过 `http://localhost:8011` 访问前端页面

如果我们在其它服务器上（例如：`*.*.*.*` ）启动这个前端服务（运行 `yarn dev --port 8011` ），我们就可以通过在游览器中访问 `http://*.*.*.*:8011` 访问前端页面

那前端跟后端是什么关系呢？ 两个是独立的，只要前端能够通过代理访问到后端的接口，那就没有问题。你可以在 A 机器上部署后端服务，然后在 B 机器上部署前端服务。我们在 `./web_client/vite.config.js` 中将 `/api` 映射到的是 `http://localhost:8010`，你可以把它配置成任意你想要访问后端地址。

当前端在以 `*.*.*.*` 这类以 IP 地址形式的网页中访问时，由于游览器的安全限制，会禁止录音，需要重新配置游览器的安全策略， 可以看下面 FAQ 部分： [【前端以IP地址的形式访问，无法录音】]


## FAQ 

#### Q: 如何安装node.js

A： node.js的安装可以参考[【菜鸟教程】](https://www.runoob.com/nodejs/nodejs-install-setup.html), 确保 npm 可用

#### Q：后端如果部署在其它机器或者别的端口如何修改

A：后端的配置地址有分散在两个文件中

修改第一个文件 `./web_client/vite.config.js`

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

修改第二个文件 `./web_client/src/api/API.js`（ Websocket 代理配置失败，所以需要在这个文件中修改）

```
// websocket （这里改成后端所在的接口）
CHAT_SOCKET_RECORD: 'ws://localhost:8010/ws/asr/offlineStream', // ChatBot websocket 接口
ASR_SOCKET_RECORD: 'ws://localhost:8010/ws/asr/onlineStream',  // Stream ASR 接口
TTS_SOCKET_RECORD: 'ws://localhost:8010/ws/tts/online', // Stream TTS 接口
```

#### Q：前端以IP地址的形式访问，无法录音

A：这里主要是游览器安全策略的限制，需要配置游览器后重启。游览器修改配置可参考[使用js-audio-recorder报浏览器不支持getUserMedia](https://blog.csdn.net/YRY_LIKE_YOU/article/details/113745273)

chrome设置地址: chrome://flags/#unsafely-treat-insecure-origin-as-secure

#### Q: npm / yarn 配置淘宝镜像源

A: 配置淘宝镜像源，详细可以参考 [【yarn npm 设置淘宝镜像】](https://www.jianshu.com/p/f6f43e8f9d6b)

```bash
# npm 配置淘宝镜像源
npm config set registry https://registry.npmmirror.com

# yarn 配置淘宝镜像源
yarn config set registry http://registry.npm.taobao.org/
```

## 参考资料

vue实现录音参考资料：https://blog.csdn.net/qq_41619796/article/details/107865602#t1

前端流式播放音频参考仓库：

https://github.com/AnthumChris/fetch-stream-audio

https://bm.enthuses.me/buffered.php?bref=6677
