# Alignment

我们首先从建模的角度理解一下对齐。语音识别任务，需要对输入音频序列 X = [x1，x2，x3...，xt...，xT] （通常是 fbank 或 mfcc 等音频特征）和输出的标注数据文本序列 Y = [y1，y2，y3...，yu...，yU] 关系进行建模，其中 X 的长度一般大于 Y 的长度。如果能够知道yu和xt的对应关系，就可以将这类任务变成语音帧级别上的分类任务，即对每个时刻 xt 进行分类得到 yu。

## MFA





## CTC Alignment





## Reference

* [ctc alignment](https://mp.weixin.qq.com/s/4aGehNN7PpIvCh03qTT5oA)
* [时间戳和N-Best](https://mp.weixin.qq.com/s?__biz=MzU2NjUwMTgxOQ==&mid=2247483956&idx=1&sn=80ce595238d84155d50f08c0d52267d3&chksm=fcaacae0cbdd43f62b1da60c8e8671a9e0bb2aeee94f58751839b03a1c45b9a3889b96705080&scene=21#wechat_redirect)
