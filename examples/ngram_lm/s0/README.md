# Ngram LM

Train chinese chararctor ngram lm by [kenlm](https://github.com/kpu/kenlm).

## Run
```
. path.sh
bash run.sh
```

## Results

```
exp/
|-- text
|-- text.char.tn
|-- text.word.tn
|-- text_zh_char_o5_p0_1_2_4_4_a22_q8_b8.arpa
|-- text_zh_char_o5_p0_1_2_4_4_a22_q8_b8.arpa.klm.bin
|-- text_zh_word_o3_p0_0_0_a22_q8_b8.arpa
`-- text_zh_word_o3_p0_0_0_a22_q8_b8.arpa.klm.bin

0 directories, 7 files
```

```
3ae083627b9b6cef1a82d574d8483f97  exp/text
d97da252d2a63a662af22f98af30cb8c  exp/text.char.tn
c18b03005bd094dbfd9b46442be361fd  exp/text.word.tn
73dbf50097896eda33985e11e1ba9a3a  exp/text_zh_char_o5_p0_1_2_4_4_a22_q8_b8.arpa
01334e2044c474b99c4f2ffbed790626  exp/text_zh_char_o5_p0_1_2_4_4_a22_q8_b8.arpa.klm.bin
36a42de548045b54662411ae7982c77f  exp/text_zh_word_o3_p0_0_0_a22_q8_b8.arpa
332422803ffd73dd7ffd16cd2b0abcd5  exp/text_zh_word_o3_p0_0_0_a22_q8_b8.arpa.klm.bin
```

```
==> exp/text <==
少先队员因该为老人让坐
祛痘印可以吗？有效果吗？
不知这款牛奶口感怎样？ 小孩子喝行吗！
是转基因油?
我家宝宝13斤用多大码的
会起坨吗？
请问给送上楼吗？
亲是送赁上门吗
送货时候有外包装没有还是直接发货过来
会不会有坏的？

==> exp/text.char.tn <==
少 先 队 员 因 该 为 老 人 让 坐
祛 痘 印 可 以 吗 有 效 果 吗
不 知 这 款 牛 奶 口 感 怎 样 小 孩 子 喝 行 吗
是 转 基 因 油
我 家 宝 宝 十 三 斤 用 多 大 码 的
会 起 坨 吗
请 问 给 送 上 楼 吗
亲 是 送 赁 上 门 吗
送 货 时 候 有 外 包 装 没 有 还 是 直 接 发 货 过 来
会 不 会 有 坏 的

==> exp/text.word.tn <==
少先队员 因该 为 老人 让 坐
祛痘 印 可以 吗 有 效果 吗
不知 这 款 牛奶 口感 怎样 小孩子 喝行 吗
是 转基因 油
我家 宝宝 十三斤 用多大码 的
会起 坨 吗
请问 给 送 上楼 吗
亲是 送赁 上门 吗
送货 时候 有 外包装 没有 还是 直接 发货 过来
会 不会 有坏 的

==> exp/text_zh_char_o5_p0_1_2_4_4_a22_q8_b8.arpa <==
\data\
ngram 1=587
ngram 2=395
ngram 3=100
ngram 4=2
ngram 5=0

\1-grams:
-3.272324       <unk>   0
0       <s>     -0.36706257

==> exp/text_zh_word_o3_p0_0_0_a22_q8_b8.arpa <==
\data\
ngram 1=689
ngram 2=1398
ngram 3=1506

\1-grams:
-3.1755018      <unk>   0
0       <s>     -0.23069073
-1.2318869      </s>    0
-3.067262       少先队员        -0.051341705
```
