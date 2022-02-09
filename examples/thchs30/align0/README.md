# THCHS-30 数据集强制对齐实验
-----
本实验对 THCHS-30 中文数据集用 [Montreal-Forced-Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html) 进行强制对齐。
THCHS-30 的文本标注数据分为：
 1. 汉字级别（word），该数据集用空格对词进行了划分，我们在使用时按照将不同字之间按空格划分
 2. 音节级别（syllable），即汉语中的一个拼音
 3. 音素级别（phone），一个拼音有多个音素组成，汉语的声母韵母可以理解为音素，不同的数据集有各自的音素标准，THCHS-30 数据集与标贝 BZNSYP 数据集的音素标准略有不同

 数据 A11_0 文本示例如下：
```
绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然↩
lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1 de5 di3 se4 si4 yue4 de5 lin2 luan2 geng4 shi4 lv4 de5 xian1 huo2 xiu4 mei4 shi1 yi4 ang4 ran2↩
l v4 sh ix4 ii iang2 ch un1 ii ian1 j ing3 d a4 k uai4 uu un2 zh ang1 d e5 d i3 s e4 s iy4 vv ve4 d e5 l in2 l uan2 g eng4 sh ix4 l v4 d e5 x ian1 h uo2 x iu4 m ei4 sh ix1 ii i4 aa ang4 r an2
```
## 开始实验
---
在本项目的 根目录/tools 执行
```
make
```
下载 MFA 的可执行包（也会同时下载本项目所需的其他工具）
执行如下命令：
```
cd a0
./run.sh
```
应用程序会自动下载 THCHS-30数据集，处理成 MFA 所需的文件格式并开始训练，您可以修改 `run.sh` 中的参数 `LEXICON_NAME` 来决定您需要强制对齐的级别（word、syllable 和 phone）
## MFA 所使用的字典
---
MFA 字典的格式请参考: [MFA 官方文档](https://montreal-forced-aligner.readthedocs.io/en/latest/)
phone.lexicon 直接使用的是 `THCHS-30/data_thchs30/lm_phone/lexicon.txt`
word.lexicon 考虑到了中文的多音字，使用**带概率的字典**, 生成规则请参考 `local/gen_word2phone.py`
`syllable.lexicon` 获取自 [DNSun/thchs30-pinyin2tone](https://github.com/DNSun/thchs30-pinyin2tone)
## 对齐结果
---
我们提供了三种级别 MFA 训练好的对齐结果、模型和字典（`syllable.lexicon`  在 `data/dict` 中，`phone.lexicon` 和` word.lexicon` 运行数据预处理代码后会自动从原始数据集复制或生成）

**phone 级别：** [phone.lexicon](https://paddlespeech.bj.bcebos.com/MFA/THCHS30/phone/phone.lexicon)、 [对齐结果](https://paddlespeech.bj.bcebos.com/MFA/THCHS30/phone/thchs30_alignment.tar.gz)、[模型](https://paddlespeech.bj.bcebos.com/MFA/THCHS30/phone/thchs30_model.zip)
**syllabel 级别：** [syllable.lexicon](https://paddlespeech.bj.bcebos.com/MFA/THCHS30/syllable/syllable.lexicon)、[对齐结果](https://paddlespeech.bj.bcebos.com/MFA/THCHS30/syllable/thchs30_alignment.tar.gz)、[模型](https://paddlespeech.bj.bcebos.com/MFA/THCHS30/syllable/thchs30_model.zip)
**word 级别：** [word.lexicon](https://paddlespeech.bj.bcebos.com/MFA/THCHS30/word/word.lexicon)、[对齐结果](https://paddlespeech.bj.bcebos.com/MFA/THCHS30/word/thchs30_alignment.tar.gz)、[模型](https://paddlespeech.bj.bcebos.com/MFA/THCHS30/word/thchs30_model.zip)

随后，您可以参考 [MFA 官方文档](https://montreal-forced-aligner.readthedocs.io/en/latest/) 使用我们给您提供好的模型直接对自己的数据集进行强制对齐，注意，您需要使用和模型对应的 lexicon 文件，当文本是汉字时，您需要用空格把不同的**汉字**（而不是词语）分开
