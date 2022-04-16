# ngram train for mandarin

Quick run:
```
bash run.sh --stage -1
```

## input

input files:
```
data/
├── lexicon.txt
├── text
└── vocab.txt
```

```
==> data/text <==
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

==> data/lexicon.txt <==
SIL sil
<SPOKEN_NOISE> sil
啊 aa a1
啊 aa a2
啊 aa a4
啊 aa a5
啊啊啊 aa a2 aa a2 aa a2
啊啊啊 aa a5 aa a5 aa a5
坐地 z uo4 d i4
坐实 z uo4 sh ix2
坐视 z uo4 sh ix4
坐稳 z uo4 uu un3
坐拥 z uo4 ii iong1
坐诊 z uo4 zh en3
坐庄 z uo4 zh uang1
坐姿 z uo4 z iy1

==> data/vocab.txt <==
<blank>
<unk>
A
B
C
D
E
龙
龚
龛
<eos>
```

## output

```
data/
├── local
│   ├── dict
│   │   ├── lexicon.txt
│   │   └── units.txt
│   └── lm
│       ├── heldout
│       ├── lm.arpa
│       ├── text
│       ├── text.no_oov
│       ├── train
│       ├── unigram.counts
│       ├── word.counts
│       └── wordlist
```

```
/workspace/srilm/bin/i686-m64/ngram-count
Namespace(bpemodel=None, in_lexicon='data/lexicon.txt', out_lexicon='data/local/dict/lexicon.txt', unit_file='data/vocab.txt')
Ignoring words 矽, which contains oov unit
Ignoring words 傩, which contains oov unit
Ignoring words 堀, which contains oov unit
Ignoring words 莼, which contains oov unit
Ignoring words 菰, which contains oov unit
Ignoring words 摭, which contains oov unit
Ignoring words 帙, which contains oov unit
Ignoring words 迨, which contains oov unit
Ignoring words 孥, which contains oov unit
Ignoring words 瑗, which contains oov unit
...
...
...
file data/local/lm/heldout: 10000 sentences, 89496 words, 0 OOVs
0 zeroprobs, logprob= -270337.9 ppl= 521.2819 ppl1= 1048.745
build LM done.
```