# Chinese Text Normalization for Speech Processing

## Problem

Search for "Text Normalization"(TN) on Google and Github, you can hardly find open-source projects that are "read-to-use" for text normalization tasks. Instead, you find a bunch of NLP toolkits or frameworks that *supports* TN functionality.  There is quite some work between "support text normalization" and "do text normalization".

## Reason

* TN is language-dependent, more or less.

    Some of TN processing methods are shared across languages, but a good TN module always involves language-specific knowledge and treatments, more or less.

* TN is task-specific.

    Even for the same language, different applications require quite different TN.

* TN is "dirty"

    Constructing and maintaining a set of TN rewrite-rules is painful, whatever toolkits and frameworks you choose.  Subtle and intrinsic complexities hide inside TN task itself, not in tools or frameworks.

* mature TN module is an asset

    Since constructing and maintaining TN is hard, it is actually an asset for commercial companies, hence it is unlikely to find a product-level TN in open-source community (correct me if you find any)

* TN is a less important topic for either academic or commercials.

## Goal

This project sets up a ready-to-use TN module for **Chinese**. Since my background is **speech processing**, this project should be able to handle most common TN tasks, in **Chinese ASR** text processing pipelines.

## Normalizers

1. supported NSW (Non-Standard-Word) Normalization

    |NSW type|raw|normalized|
    |-|-|-|
    |cardinal|这块黄金重达324.75克|这块黄金重达三百二十四点七五克|
    |date|她出生于86年8月18日，她弟弟出生于1995年3月1日|她出生于八六年八月十八日 她弟弟出生于一九九五年三月一日|
    |digit|电影中梁朝伟扮演的陈永仁的编号27149|电影中梁朝伟扮演的陈永仁的编号二七一四九|
    |fraction|现场有7/12的观众投出了赞成票|现场有十二分之七的观众投出了赞成票|
    |money|随便来几个价格12块5，34.5元，20.1万|随便来几个价格十二块五 三十四点五元 二十点一万|
    |percentage|明天有62％的概率降雨|明天有百分之六十二的概率降雨|
    |telephone|这是固话0421-33441122<br>这是手机+86 18544139121|这是固话零四二一三三四四一一二二<br>这是手机八六一八五四四一三九一二一|

    acknowledgement: the NSW normalization codes are based on [Zhiyang Zhou's work here](https://github.com/Joee1995/chn_text_norm.git)

1. punctuation removal
    
    For Chinese, it removes punctuation list collected in [Zhon](https://github.com/tsroten/zhon) project, containing
    * non-stop puncs
        ```
        '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏'
        ```
    * stop puncs
        ```
        '！？｡。'
        ```

    For English, it removes Python's `string.punctuation`

1. multilingual English word upper/lower case conversion
    since ASR/TTS lexicons usually unify English entries to uppercase or lowercase, the TN module should adapt with lexicon accordingly.

## Supported text format

1. plain text, preferably one sentence per line(most common case in ASR processing).
    ```
    今天早饭吃了没
    没吃回家吃去吧
    ...
    ```
    plain text is default format.

2. Kaldi's transcription format
    ```
    KALDI_KEY_UTT001    今天早饭吃了没
    KALDI_KEY_UTT002    没吃回家吃去吧
    ...
    ```
    TN will skip first column key section, normalize latter transcription text

    pass `--has_key` option to switch to kaldi format.

_note: All input text should be UTF-8 encoded._

## Run examples

* TN (python)

make sure you have **python3**, python2.X won't work correctly.

`sh run.sh` in `TN` dir, and compare raw text and normalized text.

* ITN (thrax)

make sure you  have **thrax** installed, and your PATH should be able to find thrax binaries.

`sh run.sh` in `ITN` dir. check Makefile for grammar dependency.

## possible future work

Since TN is a typical "done is better than perfect" module in context of ASR, and the current state is sufficient for my purpose, I probably won't update this repo frequently.

there are indeed something that needs to be improved:

* For TN, NSW normalizers in TN dir are based on regular expression, I've found some unintended matches, those pattern regexps need to be refined for more precise TN coverage.

* For ITN, extend those thrax rewriting grammars to cover more scenarios.

* Further more, nowadays commercial systems start to introduce RNN-like models into TN, and a mix of (rule-based & model-based) system is state-of-the-art.  More readings about this, look for Richard Sproat and KyleGorman's work at Google.

END
