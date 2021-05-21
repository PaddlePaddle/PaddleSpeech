# ASR Text Backend

1. [Text Segmentation](text_front_end#text segmentation)
2. Text Corrector
3. Add Punctuation
4. Text Filter



## Text Corrector

* [pycorrector](https://github.com/shibing624/pycorrector)
  本项目重点解决其中的谐音、混淆音、形似字错误、中文拼音全拼、语法错误带来的纠错任务。PS：[网友源码解读](https://zhuanlan.zhihu.com/p/138981644)
* DeepCorrection [1](https://praneethbedapudi.medium.com/deepcorrection-1-sentence-segmentation-of-unpunctuated-text-a1dbc0db4e98) [2](https://praneethbedapudi.medium.com/deepcorrection2-automatic-punctuation-restoration-ac4a837d92d9) [3](https://praneethbedapudi.medium.com/deepcorrection-3-spell-correction-and-simple-grammar-correction-d033a52bc11d)  [4](https://praneethbedapudi.medium.com/deepsegment-2-0-multilingual-text-segmentation-with-vector-alignment-fd76ce62194f)



### Question

中文文本纠错任务，常见错误类型包括：

- 谐音字词，如 配副眼睛-配副眼镜
- 混淆音字词，如 流浪织女-牛郎织女
- 字词顺序颠倒，如 伍迪艾伦-艾伦伍迪
- 字词补全，如 爱有天意-假如爱有天意
- 形似字错误，如 高梁-高粱
- 中文拼音全拼，如 xingfu-幸福
- 中文拼音缩写，如 sz-深圳
- 语法错误，如 想象难以-难以想象

当然，针对不同业务场景，这些问题并不一定全部存在。

比如输入法中需要处理前四种，搜索引擎需要处理所有类型，语音识别后文本纠错只需要处理前两种， 其中'形似字错误'主要针对五笔或者笔画手写输入等。



### Solution

#### 规则的解决思路

1. 中文纠错分为两步走，第一步是错误检测，第二步是错误纠正；
2. 错误检测部分先通过结巴中文分词器切词，由于句子中含有错别字，所以切词结果往往会有切分错误的情况，这样从字粒度和词粒度两方面检测错误， 整合这两种粒度的疑似错误结果，形成疑似错误位置候选集；
3. 错误纠正部分，是遍历所有的疑似错误位置，并使用音似、形似词典替换错误位置的词，然后通过语言模型计算句子困惑度，对所有候选集结果比较并排序，得到最优纠正词。

#### 深度模型的解决思路

1. 端到端的深度模型可以避免人工提取特征，减少人工工作量，RNN序列模型对文本任务拟合能力强，rnn_attention在英文文本纠错比赛中取得第一名成绩，证明应用效果不错；
2. CRF会计算全局最优输出节点的条件概率，对句子中特定错误类型的检测，会根据整句话判定该错误，阿里参赛2016中文语法纠错任务并取得第一名，证明应用效果不错；
3. Seq2Seq模型是使用Encoder-Decoder结构解决序列转换问题，目前在序列转换任务中（如机器翻译、对话生成、文本摘要、图像描述）使用最广泛、效果最好的模型之一；
4. BERT/ELECTRA/ERNIE/MacBERT等预训练模型强大的语言表征能力，对NLP界带来翻天覆地的改变，海量的训练数据拟合的语言模型效果无与伦比，基于其MASK掩码的特征，可以简单改造预训练模型用于纠错，加上fine-tune，效果轻松达到最优。



### 规则检测方法

- kenlm：kenlm统计语言模型工具，规则方法，语言模型纠错，利用混淆集，扩展性强

#### 错误检测

- 字粒度：语言模型困惑度（ppl）检测某字的似然概率值低于句子文本平均值，则判定该字是疑似错别字的概率大。
- 词粒度：切词后不在词典中的词是疑似错词的概率大。

#### 错误纠正

- 通过错误检测定位所有疑似错误后，取所有疑似错字的音似、形似候选词，
- 使用候选词替换，基于语言模型得到类似翻译模型的候选排序结果，得到最优纠正词。

#### 思考

1. 现在的处理手段，在词粒度的错误召回还不错，但错误纠正的准确率还有待提高，更多优质的纠错集及纠错词库会有提升。
2. 另外，现在的文本错误不再局限于字词粒度上的拼写错误，需要提高中文语法错误检测（CGED, Chinese Grammar Error Diagnosis）及纠正能力。



### Reference

* https://github.com/shibing624/pycorrector
* [基于文法模型的中文纠错系统](https://blog.csdn.net/mingzai624/article/details/82390382)
* [Norvig’s spelling corrector](http://norvig.com/spell-correct.html)
* [Chinese Spelling Error Detection and Correction Based on Language Model, Pronunciation, and Shape[Yu, 2013]](http://www.aclweb.org/anthology/W/W14/W14-6835.pdf)
* [Chinese Spelling Checker Based on Statistical Machine Translation[Chiu, 2013]](http://www.aclweb.org/anthology/O/O13/O13-1005.pdf)
* [Chinese Word Spelling Correction Based on Rule Induction[yeh, 2014]](http://aclweb.org/anthology/W14-6822)
* [Neural Language Correction with Character-Based Attention[Ziang Xie, 2016]](https://arxiv.org/pdf/1603.09727.pdf)
* [Chinese Spelling Check System Based on Tri-gram Model[Qiang Huang, 2014]](http://www.anthology.aclweb.org/W/W14/W14-6827.pdf)
* [Neural Abstractive Text Summarization with Sequence-to-Sequence Models[Tian Shi, 2018]](https://arxiv.org/abs/1812.02303)
* [基于深度学习的中文文本自动校对研究与实现[杨宗霖, 2019]](https://github.com/shibing624/pycorrector/blob/master/docs/基于深度学习的中文文本自动校对研究与实现.pdf)
* [A Sequence to Sequence Learning for Chinese Grammatical Error Correction[Hongkai Ren, 2018]](https://link.springer.com/chapter/10.1007/978-3-319-99501-4_36)
* [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB)
* [Revisiting Pre-trained Models for Chinese Natural Language Processing](https://arxiv.org/abs/2004.13922)



## Add Punctuation

* DeepCorrection [1](https://praneethbedapudi.medium.com/deepcorrection-1-sentence-segmentation-of-unpunctuated-text-a1dbc0db4e98) [2](https://praneethbedapudi.medium.com/deepcorrection2-automatic-punctuation-restoration-ac4a837d92d9) [3](https://praneethbedapudi.medium.com/deepcorrection-3-spell-correction-and-simple-grammar-correction-d033a52bc11d)  [4](https://praneethbedapudi.medium.com/deepsegment-2-0-multilingual-text-segmentation-with-vector-alignment-fd76ce62194f)



## Text Filter

* 敏感词（黄暴、涉政、违法违禁等）
