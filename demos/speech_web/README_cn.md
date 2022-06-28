# Paddle Speech Demo

PaddleSpeechDemo是一个以PaddleSpeech的语音交互功能为主体开发的Demo展示项目，用于帮助大家更好的上手PaddleSpeech以及使用PaddleSpeech构建自己的应用。

智能语音交互部分使用PaddleSpeech，对话以及信息抽取部分使用PaddleNLP，网页前端展示部分基于Vue3进行开发

主要功能：

+ 语音聊天：PaddleSpeech的语音识别能力+语音合成能力，对话部分基于PaddleNLP的闲聊功能
+ 声纹识别：PaddleSpeech的声纹识别功能展示
+ 语音识别：支持【实时语音识别】，【端到端识别】，【音频文件识别】三种模式
+ 语音合成：支持【流式合成】与【端到端合成】两种方式
+ 语音指令：基于PaddleSpeech的语音识别能力与PaddleNLP的信息抽取，实现交通费的智能报销

运行效果：

 ![效果](docs/效果展示.png)

## 安装

### 后端环境安装

```
# 安装环境
cd speech_server
pip install -r requirements.txt

# 下载 ie 模型，针对地点进行微调，效果更好，不下载的话会使用其它版本，效果没有这个好
cd source
mkdir model
cd model
wget https://bj.bcebos.com/paddlenlp/applications/speech-cmd-analysis/finetune/model_state.pdparams
```


### 前端环境安装

前端依赖node.js ，需要提前安装，确保npm可用，npm测试版本8.3.1，建议下载[官网](https://nodejs.org/en/)稳定版的node.js

```
# 进入前端目录
cd web_client

# 安装yarn，已经安装可跳过
npm install -g yarn

# 使用yarn安装前端依赖
yarn install
```


## 启动服务

### 开启后端服务

```
cd speech_server
# 默认8010端口
python main.py --port 8010
```

### 开启前端服务

```
cd web_client
yarn dev --port 8011
```

默认配置下，前端中配置的后台地址信息是localhost，确保后端服务器和打开页面的游览器在同一台机器上，不在一台机器的配置方式见下方的FAQ：【后端如果部署在其它机器或者别的端口如何修改】
## FAQ 

#### Q: 如何安装node.js

A： node.js的安装可以参考[【菜鸟教程】](https://www.runoob.com/nodejs/nodejs-install-setup.html), 确保npm可用

#### Q：后端如果部署在其它机器或者别的端口如何修改

A：后端的配置地址有分散在两个文件中

修改第一个文件`PaddleSpeechWebClient/vite.config.js`

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

修改第二个文件`PaddleSpeechWebClient/src/api/API.js`（Websocket代理配置失败，所以需要在这个文件中修改）

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
