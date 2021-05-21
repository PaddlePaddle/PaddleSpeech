汉字拼音转换工具（Python 版）
=============================

|Build| |appveyor| |Coverage| |Pypi version| |DOI|


将汉字转为拼音。可以用于汉字注音、排序、检索(`Russian translation`_) 。

基于 `hotoo/pinyin <https://github.com/hotoo/pinyin>`__ 开发。

* Documentation: http://pypinyin.rtfd.io/
* GitHub: https://github.com/mozillazg/python-pinyin
* License: MIT license
* PyPI: https://pypi.org/project/pypinyin
* Python version: 2.7, pypy, pypy3, 3.4, 3.5, 3.6, 3.7, 3.8

.. contents::


特性
----

* 根据词组智能匹配最正确的拼音。
* 支持多音字。
* 简单的繁体支持, 注音支持。
* 支持多种不同拼音/注音风格。


安装
----

.. code-block:: bash

    $ pip install pypinyin


使用示例
--------

Python 3(Python 2 下把 ``'中心'`` 替换为 ``u'中心'`` 即可):

.. code-block:: python

    >>> from pypinyin import pinyin, lazy_pinyin, Style
    >>> pinyin('中心')
    [['zhōng'], ['xīn']]
    >>> pinyin('中心', heteronym=True)  # 启用多音字模式
    [['zhōng', 'zhòng'], ['xīn']]
    >>> pinyin('中心', style=Style.FIRST_LETTER)  # 设置拼音风格
    [['z'], ['x']]
    >>> pinyin('中心', style=Style.TONE2, heteronym=True)
    [['zho1ng', 'zho4ng'], ['xi1n']]
    >>> pinyin('中心', style=Style.TONE3, heteronym=True)
    [['zhong1', 'zhong4'], ['xin1']]
    >>> pinyin('中心', style=Style.BOPOMOFO)  # 注音风格
    [['ㄓㄨㄥ'], ['ㄒㄧㄣ']]
    >>> lazy_pinyin('中心')  # 不考虑多音字的情况
    ['zhong', 'xin']


**注意事项** ：

* 拼音结果不会标明哪个韵母是轻声，轻声的韵母没有声调或数字标识（使用 ``5`` 标识轻声的方法见 `文档 <https://pypinyin.readthedocs.io/zh_CN/master/contrib.html#neutraltonewith5mixin>`__ ）。
* 无声调相关拼音风格下的结果会使用 ``v`` 表示 ``ü`` （使用 ``ü`` 代替 ``v`` 的方法见 `文档 <https://pypinyin.readthedocs.io/zh_CN/master/contrib.html#v2umixin>`__ ）。

命令行工具：

.. code-block:: console

    $ pypinyin 音乐
    yīn yuè
    $ pypinyin -h


文档
--------

详细文档请访问：http://pypinyin.rtfd.io/ 。

项目代码开发方面的问题可以看看 `开发文档`_ 。


FAQ
---------

词语中的多音字拼音有误？
+++++++++++++++++++++++++++++

目前是通过词组拼音库的方式来解决多音字问题的。如果出现拼音有误的情况，
可以自定义词组拼音来调整词语中的拼音：

.. code-block:: python

    >>> from pypinyin import Style, pinyin, load_phrases_dict
    >>> pinyin('步履蹒跚')
    [['bù'], ['lǚ'], ['mán'], ['shān']]
    >>> load_phrases_dict({'步履蹒跚': [['bù'], ['lǚ'], ['pán'], ['shān']]})
    >>> pinyin('步履蹒跚')
    [['bù'], ['lǚ'], ['pán'], ['shān']]

详见 `文档 <https://pypinyin.readthedocs.io/zh_CN/master/usage.html#custom-dict>`__ 。

为什么没有 y, w, yu 几个声母？
++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: python

    >>> from pypinyin import Style, pinyin
    >>> pinyin('下雨天', style=Style.INITIALS)
    [['x'], [''], ['t']]

因为根据 `《汉语拼音方案》 <http://www.moe.edu.cn/s78/A19/yxs_left/moe_810/s230/195802/t19580201_186000.html>`__ ，
y，w，ü (yu) 都不是声母。

    声母风格（INITIALS）下，“雨”、“我”、“圆”等汉字返回空字符串，因为根据
    `《汉语拼音方案》 <http://www.moe.edu.cn/s78/A19/yxs_left/moe_810/s230/195802/t19580201_186000.html>`__ ，
    y，w，ü (yu) 都不是声母，在某些特定韵母无声母时，才加上 y 或 w，而 ü 也有其特定规则。    —— @hotoo

    **如果你觉得这个给你带来了麻烦，那么也请小心一些无声母的汉字（如“啊”、“饿”、“按”、“昂”等）。
    这时候你也许需要的是首字母风格（FIRST_LETTER）**。    —— @hotoo

    参考: `hotoo/pinyin#57 <https://github.com/hotoo/pinyin/issues/57>`__,
    `#22 <https://github.com/mozillazg/python-pinyin/pull/22>`__,
    `#27 <https://github.com/mozillazg/python-pinyin/issues/27>`__,
    `#44 <https://github.com/mozillazg/python-pinyin/issues/44>`__

如果觉得这个行为不是你想要的，就是想把 y 当成声母的话，可以指定 ``strict=False`` ，
这个可能会符合你的预期：

.. code-block:: python

    >>> from pypinyin import Style, pinyin
    >>> pinyin('下雨天', style=Style.INITIALS)
    [['x'], [''], ['t']]
    >>> pinyin('下雨天', style=Style.INITIALS, strict=False)
    [['x'], ['y'], ['t']]

详见 `strict 参数的影响`_ 。

如何减少内存占用
++++++++++++++++++++

如果对拼音的准确性不是特别在意的话，可以通过设置环境变量 ``PYPINYIN_NO_PHRASES``
和 ``PYPINYIN_NO_DICT_COPY`` 来节省内存。
详见 `文档 <https://pypinyin.readthedocs.io/zh_CN/master/faq.html#no-phrases>`__


更多 FAQ 详见文档中的
`FAQ <https://pypinyin.readthedocs.io/zh_CN/master/faq.html>`__ 部分。


.. _#13 : https://github.com/mozillazg/python-pinyin/issues/113
.. _strict 参数的影响: https://pypinyin.readthedocs.io/zh_CN/master/usage.html#strict


拼音数据
---------

* 单个汉字的拼音使用 `pinyin-data`_ 的数据
* 词组的拼音使用 `phrase-pinyin-data`_ 的数据


Related Projects
-----------------

* `hotoo/pinyin`__: 汉字拼音转换工具 Node.js/JavaScript 版。
* `mozillazg/go-pinyin`__: 汉字拼音转换工具 Go 版。
* `mozillazg/rust-pinyin`__: 汉字拼音转换工具 Rust 版。


__ https://github.com/hotoo/pinyin
__ https://github.com/mozillazg/go-pinyin
__ https://github.com/mozillazg/rust-pinyin


.. |Build| image:: https://img.shields.io/circleci/project/github/mozillazg/python-pinyin/master.svg
   :target: https://circleci.com/gh/mozillazg/python-pinyin
.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/ni8gdyextfa85yqo/branch/master?svg=true
   :target: https://ci.appveyor.com/project/mozillazg/python-pinyin
.. |Coverage| image:: https://img.shields.io/codecov/c/github/mozillazg/python-pinyin/master.svg
   :target: https://codecov.io/gh/mozillazg/python-pinyin
.. |PyPI version| image:: https://img.shields.io/pypi/v/pypinyin.svg
   :target: https://pypi.org/project/pypinyin/
.. |DOI| image:: https://zenodo.org/badge/12830126.svg
   :target: https://zenodo.org/badge/latestdoi/12830126



.. _Russian translation: https://github.com/mozillazg/python-pinyin/blob/master/README_ru.rst
.. _pinyin-data: https://github.com/mozillazg/pinyin-data
.. _phrase-pinyin-data: https://github.com/mozillazg/phrase-pinyin-data
.. _开发文档: https://pypinyin.readthedocs.io/zh_CN/develop/develop.html



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

> Date: 2018-11-09 21:36:19 GMT [JHJ]    
> Unicode version: 12.0.0

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

[unihan]: http://www.unicode.org/charts/unihan.html


# phrase-pinyin-data [![Build Status](https://travis-ci.org/mozillazg/phrase-pinyin-data.svg?branch=master)](https://travis-ci.org/mozillazg/phrase-pinyin-data)

词语拼音数据。


## 数据介绍

拼音数据的格式：

```
{phrase}: {pinyin}
```

* 以 `#` 开头的行是注释
* 行尾的 `#` 也是注释
* `{phrase}` 汉字词语
* `{pinyin}` 词语的拼音，使用空格分隔每个汉字的拼音
* 一行一个词语的读音，有多个音的词语会出现在多行
* 示例：

  ```
  # 注释
  中国: zhōng guó
  北京: běi jīng  # 注释
  ```

文件说明:

* `overwrite.txt`: 手工纠正的拼音数据
* `pinyin.txt`: `pinyin.txt + overwrite.txt` 后的拼音数据
* `zdic_cibs.txt`: [汉典网](http://www.zdic.net/) 汉语词典拼音数据
* `zdic_cybs.txt`: [汉典网](http://www.zdic.net/) 成语词典拼音数据
* `cc_cedict.txt`: [cc-cedict.org](https://cc-cedict.org/) 拼音数据
* `large_pinyin.txt`: `zdic_cibs.txt + zdic_cybs.txt + cc_cedict.txt + pinyin.txt + overwrite.txt` 后的拼音数据


## 参考资料

* 初始数据基于 [phrases-dict.js](https://github.com/hotoo/pinyin/blob/05f74496c34ccb32db1a0fd0b358a798a22a51e5/data/phrases-dict.js) 和 [phrases_dict.py](https://github.com/mozillazg/python-pinyin/blob/366de0363ff1fb9a718ce668448bea59de09a4bf/pypinyin/phrases_dict.py)
* [汉典 zdic.net](http://www.zdic.net/)
* [字海网，叶典网](http://zisea.com/)
* [国学大师_国学网](http://www.guoxuedashi.com/)
* [CC-CEDICT download - MDBG English to Chinese dictionary](http://www.mdbg.net/chindict/chindict.php?page=cc-cedict)

