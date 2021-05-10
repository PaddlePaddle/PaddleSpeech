# pinyin-data [![Build Status](https://travis-ci.org/mozillazg/pinyin-data.svg?branch=master)](https://travis-ci.org/mozillazg/pinyin-data)

汉字拼音数据。


## 数据介绍

拼音数据的格式：

    {code point}: {pinyins}  # {hanzi} {comments}

* 以 `#` 开头的行是注释，行内 `#` 后面的字符也是注释
* `{pinyins}` 中使用逗号分隔多个拼音
* 示例：

        # 注释
        U+4E2D: zhōng,zhòng  # 中


[Unihan Database][unihan] 数据版本：

> Date: 2020-02-18 18:27:33 GMT [JHJ]     
> Unicode version: 13.0.0

* `kTGHZ2013.txt`: [Unihan Database][unihan] 中 [kTGHZ2013](http://www.unicode.org/reports/tr38/#kTGHZ2013) 部分的拼音数据（来源于《通用规范汉字字典》的拼音数据）
* `kHanyuPinyin.txt`: [Unihan Database][unihan] 中 [kHanyuPinyin](http://www.unicode.org/reports/tr38/#kHanyuPinyin) 部分的拼音数据（来源于《漢語大字典》的拼音数据）
* `kXHC1983.txt`: [Unihan Database][unihan] 中 [kXHC1983](http://www.unicode.org/reports/tr38/#kXHC1983) 部分的拼音数据（来源于《现代汉语词典》的拼音数据）
* `kHanyuPinlu.txt`: [Unihan Database][unihan] 中 [kHanyuPinlu](http://www.unicode.org/reports/tr38/#kHanyuPinlu) 部分的拼音数据（来源于《現代漢語頻率詞典》的拼音数据）
* `kMandarin.txt`: [Unihan Database][unihan] 中 [kMandarin](http://www.unicode.org/reports/tr38/#kMandarin) 部分的拼音数据（普通话中最常用的一个读音。zh-CN 为主，如果 zh-CN 中没有则使用 zh-TW 中的拼音）
* `kMandarin_overwrite.txt`: 手工纠正 `kMandarin.txt` 中有误的拼音数据（**可以修改**）
* `GBK_PUA.txt`: [Private Use Area](https://en.wikipedia.org/wiki/Private_Use_Areas) 中有拼音的汉字，参考 [GB 18030 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/GB_18030#PUA) （**可以修改**）
* `nonCJKUI.txt`: 不属于 [CJK Unified Ideograph](https://en.wikipedia.org/wiki/CJK_Unified_Ideographs) 但是却有拼音的字符（**可以修改**）
* `kanji.txt`: [日本自造汉字](https://zh.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC%E6%B1%89%E5%AD%97#7_%E6%97%A5%E6%9C%AC%E6%B1%89%E5%AD%97%E7%9A%84%E6%B1%89%E8%AF%AD%E6%99%AE%E9%80%9A%E8%AF%9D%E8%A7%84%E8%8C%83%E8%AF%BB%E9%9F%B3%E8%A1%A8) 的拼音数据 （**可以修改**）
* `kMandarin_8105.txt`: [《通用规范汉字表》](https://zh.wikipedia.org/wiki/通用规范汉字表)(2013 年版)里 8105 个汉字最常用的一个读音 (**可以修改**)
* `overwrite.txt`: 手工纠正的拼音数据（**可以修改**）
* `pinyin.txt`: 合并上述文件后的拼音数据
* `zdic.txt`: [汉典网](http://zdic.net) 的拼音数据（**可以修改**）


## 修改数据

* 上面标注了 **可以修改** 字样的文件都可以直接修改
* 如果汉字的拼音不需要修改，只是调整第一个读音的话，可以直接修改 `kMandarin_8105.txt` 这个文件
* 执行 `merge_unihan` 命令可以按照合并规则生成最新的 `pinyin.txt` 文件
* 进入 unihan 目录，执行 `make update` 命令可以更新最新的 Unihan 数据


## 参考资料

* [汉语拼音方案](http://www.moe.edu.cn/s78/A19/yxs_left/moe_810/s230/195802/t19580201_186000.html)
* [Unihan Database Lookup](http://www.unicode.org/charts/unihan.html)
* [汉典 zdic.net](http://www.zdic.net/)
* [字海网，叶典网](http://zisea.com/)
* [国学大师_国学网](http://www.guoxuedashi.com/)
* [Unicode、GB2312、GBK和GB18030中的汉字](http://www.fmddlmyy.cn/text24.html)
* [GB 18030 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/GB_18030#PUA)
* [通用规范汉字表 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8)
* [China’s 通用规范汉字表 (Tōngyòng Guīfàn Hànzìbiǎo)](https://blogs.adobe.com/CCJKType/2014/03/china-8105.html)
* [日本汉字的汉语读音规范](http://www.moe.gov.cn/s78/A19/yxs_left/moe_810/s230/201001/t20100115_75698.html)
* [日本汉字的汉语普通话规范读音表- 维基百科](https://zh.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC%E6%B1%89%E5%AD%97#7_%E6%97%A5%E6%9C%AC%E6%B1%89%E5%AD%97%E7%9A%84%E6%B1%89%E8%AF%AD%E6%99%AE%E9%80%9A%E8%AF%9D%E8%A7%84%E8%8C%83%E8%AF%BB%E9%9F%B3%E8%A1%A8)
* [漢語大字典（第二版）](http://www.ivantsoi.com/hydzd/index.html)

[unihan]: http://www.unicode.org/charts/unihan.html


## 相关项目

* [mozillazg/phrase-pinyin-data](https://github.com/mozillazg/phrase-pinyin-data): 词语拼音数据
