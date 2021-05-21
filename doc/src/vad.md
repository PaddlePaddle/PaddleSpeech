# VAD

## Endpoint Detection

### Kaldi

 **Kaldi**使用规则方式，制定了五条规则，只要满足其中一条则认为是检测到了 endpoint。

1. 识别出文字之前，检测到了 5s 的静音；
2. 识别出文字之后，检测到了 2s 的静音；
3. 解码到概率较小的 final state，且检测到了 1s 的静音；
4. 解码到概率较大的 final state，且检测到了 0.5s 的静音；
5. 已经解码了 20s。

### CTC

将连续的长 blank 标签，视为非语音区域。非语音区域满足一定的条件，即可认为是检测到了 endpoint。同时，参考 Kaldi 的 `src/online2/online-endpoint.h`，制定了以下三条规则：

1. 识别出文字之前，检测到了 5s 的静音；
2. 识别出文字之后，检测到了 1s 的静音；
3. 已经解码了 20s。

只要满足上述三条规则中的任意一条， 就认为检测到了 endpoint。



## Reference

* [Endpoint 检测](https://mp.weixin.qq.com/s?__biz=MzU2NjUwMTgxOQ==&mid=2247484024&idx=1&sn=12da2ee76347de4a18856274ba6ba61f&chksm=fcaacaaccbdd43ba6b3e996bbf1e2ac6d5f1b449dfd80fcaccfbbe0a240fa1668b931dbf4bd5&scene=21#wechat_redirect)
* Kaldi: *https://github.com/kaldi-asr/kaldi/blob/6260b27d146e466c7e1e5c60858e8da9fd9c78ae/src/online2/online-endpoint.h#L132-L150*
* End-to-End Automatic Speech Recognition Integrated with CTC-Based Voice Activity Detection: *https://arxiv.org/pdf/2002.00551.pdf*
