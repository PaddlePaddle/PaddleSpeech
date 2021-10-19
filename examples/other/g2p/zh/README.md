# G2P

* WS
jieba
* G2P
pypinyin
* Tone sandhi
simple

We recommend using [Paraket](https://github.com/PaddlePaddle/Parakeet] [TextFrontEnd](https://github.com/PaddlePaddle/Parakeet/blob/develop/parakeet/frontend/__init__.py) to do G2P.
The phoneme set should be changed, you can reference `examples/thchs30/a0/data/dict/syllable.lexicon`.

## Download Baker dataset

[Baker](https://test.data-baker.com/#/data/index/source) dataset has to be downloaded mannually and moved to './data',
because you will have to pass the `CATTCHA` from a browswe to download the dataset.


## RUN

```
. path.sh
./run.sh
```

## Result

```
exp/
|-- 000001-010000.txt
|-- ref.pinyin
|-- trans.jieba.pinyin
`-- trans.pinyin

0 directories, 4 files
```

```
4f5a368441eb16aaf43dc1972f8b63dd  exp/000001-010000.txt
01707896391c2de9b6fc4a39654be942  exp/ref.pinyin
43380ef160f65a23a3a0544700aa49b8  exp/trans.jieba.pinyin
8e6ff1fc22d8e8584082e804e8bcdeb7  exp/trans.pinyin
```

```
==> exp/000001-010000.txt <==
000001  卡尔普#2陪外孙#1玩滑梯#4。
        ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
000002  假语村言#2别再#1拥抱我#4。
        jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
000003  宝马#1配挂#1跛骡鞍#3，貂蝉#1怨枕#2董翁榻#4。
        bao2 ma3 pei4 gua4 bo3 luo2 an1 diao1 chan2 yuan4 zhen3 dong3 weng1 ta4
000004  邓小平#2与#1撒切尔#2会晤#4。
        deng4 xiao3 ping2 yu3 sa4 qie4 er3 hui4 wu4
000005  老虎#1幼崽#2与#1宠物犬#1玩耍#4。
        lao2 hu3 you4 zai3 yu2 chong3 wu4 quan3 wan2 shua3

==> exp/ref.pinyin <==
000001 ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
000002 jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
000003 bao2 ma3 pei4 gua4 bo3 luo2 an1 diao1 chan2 yuan4 zhen3 dong3 weng1 ta4
000004 deng4 xiao3 ping2 yu3 sa4 qie4 er3 hui4 wu4
000005 lao2 hu3 you4 zai3 yu2 chong3 wu4 quan3 wan2 shua3
000006 shen1 chang2 yue1 wu2 chi3 er4 cun4 wu3 fen1 huo4 yi3 shang4
000007 zhao4 di2 yue1 cao2 yun2 teng2 qu4 gui3 wu1
000008 zhan2 pin3 sui1 you3 zhan3 yuan2 que4 tui2
000009 yi2 san3 ju1 er2 tong2 he2 you4 tuo1 er2 tong2 wei2 zhu3
000010 ke1 te4 ni1 shen1 chuan1 bao4 wen2 da4 yi1

==> exp/trans.jieba.pinyin <==
000001 ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1
000002 jia3 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
000003 bao3 ma3 pei4 gua4 bo3 luo2 an1 diao1 chan2 yuan4 zhen3 dong3 weng1 ta4
000004 deng4 xiao3 ping2 yu3 sa1 qie4 er3 hui4 wu4
000005 lao3 hu3 you4 zai3 yu3 chong3 wu4 quan3 wan2 shua3
000006 shen1 chang2 yue1 wu3 chi3 er4 cun4 wu3 fen1 huo4 yi3 shang4
000007 zhao4 di2 yue1 cao2 yun2 teng2 qu4 gui3 wu1
000008 zhan3 pin3 sui1 you3 zhan3 yuan2 que4 tui2
000009 yi3 san3 ju1 er2 tong2 he2 you4 tuo1 er2 tong2 wei2 zhu3
000010 ke1 te4 ni1 shen1 chuan1 bao4 wen2 da4 yi1

==> exp/trans.pinyin <==
000001 ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1
000002 jia3 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
000003 bao3 ma3 pei4 gua4 bo3 luo2 an1 diao1 chan2 yuan4 zhen3 dong3 weng1 ta4
000004 deng4 xiao3 ping2 yu3 sa1 qie4 er3 hui4 wu4
000005 lao3 hu3 you4 zai3 yu3 chong3 wu4 quan3 wan2 shua3
000006 shen1 chang2 yue1 wu3 chi3 er4 cun4 wu3 fen1 huo4 yi3 shang4
000007 zhao4 di2 yue1 cao2 yun2 teng2 qu4 gui3 wu1
000008 zhan3 pin3 sui1 you3 zhan3 yuan2 que4 tui2
000009 yi3 san3 ju1 er2 tong2 he2 you4 tuo1 er2 tong2 wei2 zhu3
000010 ke1 te4 ni1 shen1 chuan1 bao4 wen2 da4 yi1
```
