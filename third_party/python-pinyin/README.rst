汉字拼音转换工具（Python 版）
=============================

|Build| |GitHubAction| |Coverage| |Pypi version| |DOI|


将汉字转为拼音。可以用于汉字注音、排序、检索(`Russian translation`_) 。

基于 `hotoo/pinyin <https://github.com/hotoo/pinyin>`__ 开发。

* Documentation: http://pypinyin.rtfd.io/
* GitHub: https://github.com/mozillazg/python-pinyin
* License: MIT license
* PyPI: https://pypi.org/project/pypinyin
* Python version: 2.7, pypy, pypy3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9

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
    >>> lazy_pinyin('战略', v_to_u=True)  # 不使用 v 表示 ü
    ['zhan', 'lüe']
    # 使用 5 标识轻声
    >>> lazy_pinyin('衣裳', style=Style.TONE3, neutral_tone_with_five=True)
    ['yi1', 'shang5']


**注意事项** ：

* 默认情况下拼音结果不会标明哪个韵母是轻声，轻声的韵母没有声调或数字标识（可以通过参数 ``neutral_tone_with_five=True`` 开启使用 ``5`` 标识轻声 ）。
* 默认情况下无声调相关拼音风格下的结果会使用 ``v`` 表示 ``ü`` （可以通过参数 ``v_to_u=True`` 开启使用 ``ü`` 代替 ``v`` ）。
* 默认情况下会原样输出没有拼音的字符（自定义处理没有拼音的字符的方法见 `文档 <https://pypinyin.readthedocs.io/zh_CN/master/usage.html#handle-no-pinyin>`__ ）。

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

如果是分词导致的拼音有误的话，可以先使用其他的分词模块对数据进行分词处理，
然后将分词后的词组结果列表作为函数的参数即可:

.. code-block:: python

    >>> # 使用其他分词模块分词，比如 jieba 之类，
    >>> #或者基于 phrases_dict.py 里的词语数据使用其他分词算法分词
    >>> words = list(jieba.cut('每股24.67美元的确定性协议'))
    >>> pinyin(words)


为什么没有 y, w, yu 几个声母？
++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: python

    >>> from pypinyin import Style, pinyin
    >>> pinyin('下雨天', style=Style.INITIALS)
    [['x'], [''], ['t']]

因为根据 `《汉语拼音方案》 <http://www.moe.gov.cn/s78/A19/yxs_left/moe_810/s230/195802/t19580201_186000.html>`__ ，
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
* 声母和韵母使用 `《汉语拼音方案》 <http://www.moe.gov.cn/s78/A19/yxs_left/moe_810/s230/195802/t19580201_186000.html>`__ 的数据


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
.. |GitHubAction| image:: https://github.com/mozillazg/python-pinyin/workflows/CI/badge.svg
   :target: https://github.com/mozillazg/python-pinyin/actions
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
