# [Aishell1](http://openslr.elda.org/33/)

This Open Source Mandarin Speech Corpus, AISHELL-ASR0009-OS1, is 178 hours long. It is a part of AISHELL-ASR0009, of which utterance contains 11 domains, including smart home, autonomous driving, and industrial production. The whole recording was put in quiet indoor environment, using 3 different devices at the same time: high fidelity microphone (44.1kHz, 16-bit,); Android-system mobile phone (16kHz, 16-bit), iOS-system mobile phone (16kHz, 16-bit). Audios in high fidelity were re-sampled to 16kHz to build AISHELL- ASR0009-OS1. 400 speakers from different accent areas in China were invited to participate in the recording. The manual transcription accuracy rate is above 95%, through professional speech annotation and strict quality inspection. The corpus is divided into training, development and testing sets. ( This database is free for academic research, not in the commerce, if without permission. )


## Dataset Architecture

```bash
data_aishell
├── transcript      # text 目录
└── wav             # wav 目录
    ├── dev         # dev 目录
    │   ├── S0724   # spk 目录
    │   ├── S0725
    │   ├── S0726
    ├── train
    │   ├── S0724
    │   ├── S0725
    │   ├── S0726
    ├── test
    │   ├── S0724
    │   ├── S0725
    │   ├── S0726
 

data_aishell
├── transcript
│   └── aishell_transcript_v0.8.txt   # 文本标注文件
└── wav
    ├── dev
    │   ├── S0724
    │   │   ├── BAC009S0724W0121.wav  # S0724 的音频
    │   │   ├── BAC009S0724W0122.wav
    │   │   ├── BAC009S0724W0123.wav
    ├── test
    │   ├── S0724
    │   │   ├── BAC009S0724W0121.wav
    │   │   ├── BAC009S0724W0122.wav
    │   │   ├── BAC009S0724W0123.wav
    ├── train
    │   ├── S0724
    │   │   ├── BAC009S0724W0121.wav
    │   │   ├── BAC009S0724W0122.wav
    │   │   ├── BAC009S0724W0123.wav
    
标注文件格式： <utt> <tokens>
> head data_aishell/transcript/aishell_transcript_v0.8.txt 
BAC009S0002W0122 而 对 楼市 成交 抑制 作用 最 大 的 限 购
BAC009S0002W0123 也 成为 地方 政府 的 眼中 钉
BAC009S0002W0124 自 六月 底 呼和浩特 市 率先 宣布 取消 限 购 后
BAC009S0002W0125 各地 政府 便 纷纷 跟进
BAC009S0002W0126 仅 一 个 多 月 的 时间 里
BAC009S0002W0127 除了 北京 上海 广州 深圳 四 个 一 线 城市 和 三亚 之外
BAC009S0002W0128 四十六 个 限 购 城市 当中
BAC009S0002W0129 四十一 个 已 正式 取消 或 变相 放松 了 限 购
BAC009S0002W0130 财政 金融 政策 紧随 其后 而来
BAC009S0002W0131 显示 出 了 极 强 的 威力
```
