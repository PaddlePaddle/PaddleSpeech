# Built TLG wfst

## Input
```
data/local/
├── dict
│   ├── lexicon.txt
│   └── units.txt
└── lm
    ├── heldout
    ├── lm.arpa
    ├── text
    ├── text.no_oov
    ├── train
    ├── unigram.counts
    ├── word.counts
    └── wordlist
```

```
==> data/local/dict/lexicon.txt <==
啊 啊
啊啊啊 啊 啊 啊
阿 阿
阿尔 阿 尔
阿根廷 阿 根 廷
阿九 阿 九
阿克 阿 克
阿拉伯数字 阿 拉 伯 数 字
阿拉法特 阿 拉 法 特
阿拉木图 阿 拉 木 图

==> data/local/dict/units.txt <==
<blank>
<unk>
A
B
C
D
E
F
G
H

==> data/local/lm/heldout <==
而 对 楼市 成交 抑制 作用 最 大 的 限 购
也 成为 地方 政府 的 眼中 钉
自 六月 底 呼和浩特 市 率先 宣布 取消 限 购 后
各地 政府 便 纷纷 跟进
仅 一 个 多 月 的 时间 里
除了 北京 上海 广州 深圳 四 个 一 线 城市 和 三亚 之外
四十六 个 限 购 城市 当中
四十一 个 已 正式 取消 或 变相 放松 了 限 购
财政 金融 政策 紧随 其后 而来
显示 出 了 极 强 的 威力

==> data/local/lm/lm.arpa <==

\data\
ngram 1=129356
ngram 2=504661
ngram 3=123455

\1-grams:
-1.531278       </s>
-3.828829       <SPOKEN_NOISE>  -0.1600094
-6.157292       <UNK>

==> data/local/lm/text <==
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

==> data/local/lm/text.no_oov <==
<SPOKEN_NOISE> 而 对 楼市 成交 抑制 作用 最 大 的 限 购 
<SPOKEN_NOISE> 也 成为 地方 政府 的 眼中 钉 
<SPOKEN_NOISE> 自 六月 底 呼和浩特 市 率先 宣布 取消 限 购 后 
<SPOKEN_NOISE> 各地 政府 便 纷纷 跟进 
<SPOKEN_NOISE> 仅 一 个 多 月 的 时间 里 
<SPOKEN_NOISE> 除了 北京 上海 广州 深圳 四 个 一 线 城市 和 三亚 之外 
<SPOKEN_NOISE> 四十六 个 限 购 城市 当中 
<SPOKEN_NOISE> 四十一 个 已 正式 取消 或 变相 放松 了 限 购 
<SPOKEN_NOISE> 财政 ���融 政策 紧随 其后 而来 
<SPOKEN_NOISE> 显示 出 了 极 强 的 威力 

==> data/local/lm/train <==
汉莎 不 得 不 通过 这样 的 方式 寻求 新 的 发展 点
并 计划 朝云 计算 方面 发展
汉莎 的 基础 设施 部门 拥有 一千四百 名 员工
媒体 就 曾 披露 这笔 交易
虽然 双方 已经 正式 签署 了 外包 协议
但是 这笔 交易 还 需要 得到 反 垄断 部门 的 批准
陈 黎明 一九八九 年 获得 美国 康乃尔 大学 硕士 学位
并 于 二零零三 年 顺利 完成 美国 哈佛 商学 院 高级 管理 课程
曾 在 多家 国际 公司 任职
拥有 业务 开发 商务 及 企业 治理

==> data/local/lm/unigram.counts <==
  57487 的
  13099 在
  11862 一
  11397 了
  10998 不
   9913 是
   7952 有
   6250 和
   6152 个
   5422 将

==> data/local/lm/word.counts <==
  57486 的
  13098 在
  11861 一
  11396 了
  10997 不
   9912 是
   7951 有
   6249 和
   6151 个
   5421 将

==> data/local/lm/wordlist <==
的
在
一
了
不
是
有
和
个
将
```

## Output

```
fstaddselfloops 'echo 4234 |' 'echo 123660 |' 
Lexicon and Token FSTs compiling succeeded
arpa2fst --read-symbol-table=data/lang_test/words.txt --keep-symbols=true - 
LOG (arpa2fst[5.5.0~1-5a37]:Read():arpa-file-parser.cc:94) Reading \data\ section.
LOG (arpa2fst[5.5.0~1-5a37]:Read():arpa-file-parser.cc:149) Reading \1-grams: section.
LOG (arpa2fst[5.5.0~1-5a37]:Read():arpa-file-parser.cc:149) Reading \2-grams: section.
LOG (arpa2fst[5.5.0~1-5a37]:Read():arpa-file-parser.cc:149) Reading \3-grams: section.
Checking how stochastic G is (the first of these numbers should be small):
fstisstochastic data/lang_test/G.fst 
0 -1.14386
fsttablecompose data/lang_test/L.fst data/lang_test/G.fst 
fstminimizeencoded 
fstdeterminizestar --use-log=true 
fsttablecompose data/lang_test/T.fst data/lang_test/LG.fst 
Composing decoding graph TLG.fst succeeded
Aishell build TLG done.
```

```
data/
├── lang_test
│   ├── G.fst
│   ├── L.fst
│   ├── LG.fst
│   ├── T.fst
│   ├── TLG.fst
│   ├── tokens.txt
│   ├── units.txt
│   └── words.txt
└── local
    ├── lang
    │   ├── L.fst
    │   ├── T.fst
    │   ├── tokens.txt
    │   ├── units.txt
    │   └── words.txt
    └── tmp
        ├── disambig.list
        ├── lexiconp_disambig.txt
        ├── lexiconp.txt
        └── units.list
```
