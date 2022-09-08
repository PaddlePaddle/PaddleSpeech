# 接口文档

开启服务后可参照：

http://0.0.0.0:8010/docs

## ASR

### 【POST】/asr/offline

说明：上传 16k, 16bit wav 文件，返回 offline 语音识别模型识别结果

返回: JSON

前端接口： ASR-端到端识别，音频文件识别；语音指令-录音上传

示例:

```json
{
  "code": 0,
  "result": "你也喜欢这个天气吗",
  "message": "ok"
}
```

### 【POST】/asr/offlinefile

说明：上传16k,16bit wav文件，返回 offline 语音识别模型识别结果 + wav 数据的 base64

返回: JSON

前端接口： 音频文件识别(播放这段base64还原后记得添加 wav 头，采样率 16k, int16，添加后才能播放)

示例:

```json
{
  "code": 0,
  "result": {
    "asr_result": "今天天气真好",
    "wav_base64": "///+//3//f/8/////v/////////////////+/wAA//8AAAEAAQACAAIAAQABAP"
    },
  "message": "ok"
}
```


### 【POST】/asr/collectEnv

说明： 通过采集环境噪音，上传 16k, int16 wav 文件，来生成后台 VAD 的能量阈值， 返回阈值结果

前端接口：ASR-环境采样

返回: JSON

```json
{
  "code": 0,
  "result": 3624.93505859375,
  "message": "采集环境噪音成功"
}
```

### 【GET】/asr/stopRecord

说明：通过 GET 请求 /asr/stopRecord, 后台停止接收 offlineStream 中通过 WS 协议 上传的数据

前端接口：语音聊天-暂停录音（获取 NLP，播放 TTS 时暂停）

返回: JSON

```JSON
{
  "code": 0,
  "result": null,
  "message": "停止成功"
}
```

### 【GET】/asr/resumeRecord

说明：通过 GET 请求 /asr/resumeRecord, 后台停止接收 offlineStream 中通过 WS 协议 上传的数据

前端接口：语音聊天-恢复录音（ TTS 播放完毕时，告诉后台恢复录音）

返回: JSON

```JSON
{
  "code": 0,
  "result": null,
  "message": "Online录音恢复"
}
```

### 【Websocket】/ws/asr/offlineStream

说明：通过 WS 协议，将前端音频持续上传到后台，前端采集 16k，Int16 类型的PCM片段，持续上传到后端

前端接口：语音聊天-开始录音，持续将麦克风语音传给后端，后端推送语音识别结果

返回：后端返回识别结果，offline 模型识别结果， 由WS推送


### 【Websocket】/ws/asr/onlineStream

说明：通过 WS 协议，将前端音频持续上传到后台，前端采集 16k，Int16 类型的 PCM 片段，持续上传到后端

前端接口：ASR-流式识别开始录音，持续将麦克风语音传给后端，后端推送语音识别结果

返回：后端返回识别结果，online 模型识别结果， 由 WS 推送

## NLP

### 【POST】/nlp/chat

说明：返回闲聊对话的结果

前端接口：语音聊天-获取到ASR识别结果后，向后端获取闲聊文本

上传示例：

```json
{
  "chat": "天气非常棒"
}
```

返回示例：

```json
{
  "code": 0,
  "result": "是的,我也挺喜欢的",
  "message": "ok"
}
```


### 【POST】/nlp/ie

说明：返回信息抽取结果

前端接口：语音指令-向后端获取信息抽取结果

上传示例：

```json
{
  "chat": "今天我从马来西亚出发去香港花了五十万元"
}
```

返回示例：

```json
{
  "code": 0,
  "result": [
    {
      "时间": [
        {
          "text": "今天",
          "start": 0,
          "end": 2,
          "probability": 0.9817976247505698
        }
      ],
      "出发地": [
        {
          "text": "马来西亚",
          "start": 4,
          "end": 8,
          "probability": 0.974892389414169
        }
      ],
      "目的地": [
        {
          "text": "马来西亚",
          "start": 4,
          "end": 8,
          "probability": 0.7347504438136951
        }
      ],
      "费用": [
        {
          "text": "五十万元",
          "start": 15,
          "end": 19,
          "probability": 0.9679076530644402
        }
      ]
    }
  ],
  "message": "ok"
}
```


## TTS

### 【POST】/tts/offline

说明：获取 TTS 离线模型音频

前端接口：TTS-端到端合成

上传示例：

```json
{
  "text": "天气非常棒"
}
```

返回示例：对应音频对应的 base64 编码

```json
{
  "code": 0,
  "result": "UklGRrzQAABXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YZjQAAADAP7/BAADAAAA...",
  "message": "ok"
}
```

### 【POST】/tts/online

说明：流式获取语音合成音频

前端接口：流式合成

上传示例：
```json
{
  "text": "天气非常棒"
}

```

返回示例：

二进制PCM片段，16k Int 16类型

## VPR

### 【POST】/vpr/enroll

说明：声纹注册，通过表单上传 spk_id（字符串，非空）, 与 audio (文件)

前端接口：声纹识别-声纹注册

上传示例：

```text
curl -X 'POST' \
  'http://0.0.0.0:8010/vpr/enroll' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'spk_id=啦啦啦啦' \
  -F 'audio=@demo_16k.wav;type=audio/wav'
```

返回示例：

```json
{
  "status": true,
  "msg": "Successfully enroll data!"
}
```

### 【POST】/vpr/recog

说明：声纹识别，识别文件，提取文件的声纹信息做比对 音频 16k, int 16 wav 格式

前端接口：声纹识别-上传音频，返回声纹识别结果

上传示例： 

```shell
curl -X 'POST' \
  'http://0.0.0.0:8010/vpr/recog' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'audio=@demo_16k.wav;type=audio/wav'
```

返回示例：

```json
[
  [
    "啦啦啦啦",
    [
      "",
      100
    ]
  ],
  [
    "test1",
    [
      "",
      11.64
    ]
  ],
  [
    "test2",
    [
      "",
      6.09
    ]
  ]
]

```


### 【POST】/vpr/del

说明： 根据 spk_id 删除用户数据

前端接口：声纹识别-删除用户数据

上传示例：
```json
{
 "spk_id":"啦啦啦啦"
}
```

返回示例

```json
{
  "status": true,
  "msg": "Successfully delete data!"
}

```


### 【GET】/vpr/list

说明：查询用户列表数据，无需参数，返回 spk_id 与 vpr_id

前端接口：声纹识别-获取声纹数据列表

返回示例：

```json
[
  [
    "test1",
    "test2"
  ],
  [
    9,
    10
  ]
]

```


### 【GET】/vpr/data

说明： 根据 vpr_id 获取用户vpr时使用的音频

前端接口：声纹识别-获取vpr对应的音频

访问示例：

```shell
curl -X 'GET' \
  'http://0.0.0.0:8010/vpr/data?vprId=9' \
  -H 'accept: application/json'
```

返回示例：

对应音频文件

### 【GET】/vpr/database64

说明： 根据 vpr_id 获取用户 vpr 时注册使用音频转换成 16k, int16 类型的数组，返回 base64 编码

前端接口：声纹识别-获取 vpr 对应的音频（注意：播放时需要添加 wav头，16k,int16, 可参考 tts 播放时添加 wav 的方式，注意更改采样率）

访问示例：

```shell
curl -X 'GET' \
  'http://localhost:8010/vpr/database64?vprId=12' \
  -H 'accept: application/json'
```

返回示例：
```json
{
  "code": 0,
  "result":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "message": "ok"
```
