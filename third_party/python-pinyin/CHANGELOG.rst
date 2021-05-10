Changelog
---------

`0.41.0`_ (2021-03-13)
++++++++++++++++++++++++

* **[New]** 新增 ``pypinyin.contrib.tone_convert`` 模块，用于
  ``Style.TONE`` 、 ``Style.TONE2`` 、 ``Style.TONE3`` 、 ``Style.NORMAL`` 风格的拼音之间互相转换。
  详见 `文档 <https://pypinyin.readthedocs.io/zh_CN/develop/contrib.html#tone-convert>`__
* **[Improved]** 使用 `pinyin-data`_ v0.10.2 的拼音数据。


`0.40.0`_ (2020-11-22)
++++++++++++++++++++++++

* **[Improved]** 精简 phrases_dict, 删除 phrases_dict 中凡是能通过 pinyin_dict 得到相同结果的数据。
* **[Improved]** 使用 `phrase-pinyin-data`_ v0.10.5 的词语拼音数据。
* **[Improved]** 使用 `pinyin-data`_ v0.10.1 的拼音数据。


`0.39.1`_ (2020-10-08)
++++++++++++++++++++++++

* **[Improved]** 使用 `phrase-pinyin-data`_ v0.10.4 的词语拼音数据。
* **[Improved]** 使用 `pinyin-data`_ v0.10.0 的拼音数据。


`0.39.0`_ (2020-08-16)
++++++++++++++++++++++++

* **[New]** ``pinyin`` 和 ``lazy_pinyin`` 函数增加参数 ``v_to_u`` 和 ``neutral_tone_with_five``:

  * ``v_to_u=True`` 时在无声调相关拼音风格下使用 ``ü`` 代替原来的 ``v``

  .. code-block:: python

      >>> lazy_pinyin('战略')
      ['zhan', 'lve']
      >>> lazy_pinyin('战略', v_to_u=True)
      ['zhan', 'lüe']

  * ``neutral_tone_with_five=True`` 时在数字标识声调相关风格下使用 ``5`` 标识轻声

  .. code-block:: python

      >>> lazy_pinyin('衣裳', style=Style.TONE3)
      ['yi1', 'shang']
      >>> lazy_pinyin('衣裳', style=Style.TONE3, neutral_tone_with_five=True)
      ['yi1', 'shang5']



`0.38.1`_ (2020-07-05)
++++++++++++++++++++++++

* **[Improved]** 优化内置分词，处理前缀匹配导致无法正确识别尾部词语的问题。 Fixed `#205`_
* **[Improved]** 使用 `phrase-pinyin-data`_ v0.10.3 的词语拼音数据。


`0.38.0`_ (2020-06-07)
++++++++++++++++++++++++

* **[Improved]** 优化内置分词，严格按照是否是词语来分词。 Fixed `#139`_
* **[Improved]** 使用 `pinyin-data`_ v0.9.0 的拼音数据。


`0.37.0`_ (2020-02-09)
++++++++++++++++++++++++

* **[Bugfixed]** 修复 ``NeutralToneWith5Mixin`` 在 ``TONE3`` 相关风格未把 5 标在预期的拼音末尾位置。
* **[New]** 增加 Python 3.8 下的测试，正式支持 Python 3.8 。


`0.36.0`_ (2019-10-27)
+++++++++++++++++++++++

* **[New]** 增加 ``V2UMixin`` 用于支持无声调相关拼音风格下的结果使用 ``ü`` 代替原来的 ``v`` 。
  详见 `文档 <https://pypinyin.readthedocs.io/zh_CN/master/contrib.html#v2umixin>`__ 。
* **[New]** 增加 ``NeutralToneWith5Mixin`` 用于支持使用数字表示声调的拼音风格下使用 5 标识轻声。
  详见 `文档 <https://pypinyin.readthedocs.io/zh_CN/master/contrib.html#neutraltonewith5mixin>`__ 。
* **[New]** 增加 ``Pinyin`` 和 ``DefaultConverter`` 类用于实现自定义处理过程和结果
  （实验性功能，绝大部分用户无需关心新增的这两个类）。
* **[Improved]** 使用 `phrase-pinyin-data`_ v0.10.2 的词语拼音数据。
* **[Improved]** 使用 `pinyin-data`_ v0.8.1 的拼音数据。


`0.35.4`_ (2019-07-13)
+++++++++++++++++++++++

* **[Bugfixed]** 修复 ``m̄`` ``ê̄``  ``ế`` ``ê̌`` ``ề`` 这几个音无法转换为不含声调结果的问题。
* **[Improved]** 使用 `phrase-pinyin-data`_ v0.10.1 的词语拼音数据。 Fixed `#174`_
* **[Improved]** 使用 `pinyin-data`_ v0.8.0 的拼音数据。
* **[Improved]** 修复一处参数注释错误。(via `#176`_ Thanks `@yangwe1`_)


`0.35.3`_ (2019-05-11)
++++++++++++++++++++++++

* **[Bugfixed]** 修复鼻音 ``m̀`` 无法转换为不含声调结果的问题。
* **[Improved]** 使用 `phrase-pinyin-data`_ v0.10.0 的词语拼音数据。
  Fixed `#166`_ `#167`_ `#169`_ `#170`_
* **[Improved]** Windows CI 增加在 x64 下跑测试 (via `#164`_ Thanks `@hanabi1224`_)


`0.35.2`_ (2019-04-06)
+++++++++++++++++++++++

* **[Improved]** 使用 `phrase-pinyin-data`_ v0.9.2 的词语拼音数据。 Fixed `#159`_ `#160`_
* **[Improved]** 使用 `pinyin-data`_ v0.7.0 的拼音数据。


`0.35.1`_ (2019-03-02)
+++++++++++++++++++++++

* **[Bugfixed]** 修复 ``朝阳`` 在 ``heteronym=False`` 时输出了多个音的情况。


`0.35.0`_ (2019-02-24)
+++++++++++++++++++++++

* **[Improved]** 使用 `phrase-pinyin-data`_ v0.9.0 的词语拼音数据。 Fixed `#154`_ `#149`_
* **[New]** 支持 ``朝阳`` 这种一个词多个音（ ``'朝阳': [['zhāo', 'cháo'], ['yáng']]`` ）在多音字模式下输出多个音。 Fixed `#154`_


`0.34.1`_ (2018-12-30)
+++++++++++++++++++++++

* **[Improved]** 使用 `phrase-pinyin-data`_ v0.8.5 的词语拼音数据。 Fixed `#151`_


`0.34.0`_ (2018-12-08)
+++++++++++++++++++++++

不兼容旧版的变更
~~~~~~~~~~~~~~~~~~

* **[Changed]** 当 ``errors`` 参数的值是个回调对象并且返回值是个 ``list`` 时，
  会使用这个 list 来 extend 结果 list (via `#147`_ . Thanks `@howl-anderson`_ ) ::

    # 更新前
    >>> pinyin('你好☆☆', errors=lambda x: ['star' for _ in x])
    [['nǐ'], ['hǎo'], ['star', 'star']]

    # 更新后
    >>> pinyin('你好☆☆', errors=lambda x: ['star' for _ in x])
    [['nǐ'], ['hǎo'], ['star'], ['star']]


详见文档: https://pypinyin.readthedocs.io/zh_CN/develop/usage.html#handle-no-pinyin


`0.33.2`_ (2018-11-03)
++++++++++++++++++++++++

* **[Bugfixed]** 修复 ``strict=True`` 时韵母相关风格下没有正确处理韵母 ``üan`` 的问题。


`0.33.1`_ (2018-09-23)
++++++++++++++++++++++++

* **[Improved]** 使用 `pinyin-data`_ v0.6.2 的拼音数据。
* **[Improved]** 使用 `phrase-pinyin-data`_ v0.8.4 的词语拼音数据。


`0.33.0`_ (2018-08-05)
++++++++++++++++++++++++

* **[Bugfixed]** 修复命令行程序在 ``sys.stdin.encoding`` 为 ``None`` 时无法正常工作的问题。
* **[Improved]** 使用 `pinyin-data`_ v0.6.1 的拼音数据。
* **[Improved]** 使用 `phrase-pinyin-data`_ v0.8.3 的词语拼音数据。

  * Fixed `#137`_

* **[Changed]** 不再测试 Python 2.6 和 Python 3.3，增加测试 Python 3.7 和 PyPy3
  即不保证程序兼容 Python 2.6 和 Python 3.3。


`0.32.0`_ (2018-07-28)
++++++++++++++++++++++++

* **[Improved]** 使用 `pinyin-data`_ v0.6.0 的拼音数据。
* **[Improved]** 使用 `phrase-pinyin-data`_ v0.8.2 的词语拼音数据。


`0.31.0`_ (2018-06-10)
++++++++++++++++++++++++

* **[New]** 增加 py.typed 标记文件，支持 `PEP 561`_ (via `#130`_)
* **[Changed]** 使用 `phrase-pinyin-data`_ v0.7.3 的词语拼音数据。

  * fixed `#112`_ `#117`_ `#122`_ `#131`_
  * 精简词组拼音，删除部分数据有误的拼音数据


`0.30.1`_ (2018-04-25)
++++++++++++++++++++++++

* **[Improved]** 更新文档和测试。(via `7fa0b87 <https://github.com/mozillazg/python-pinyin/commit/7fa0b879df47e8a7e5af5edb5f243dd4ea645410>`_)
* **[Improved]** 对用户传入的已进行分词处理的数据进行二次分词以便提高准确性。(via `#126`_)
* **[Improved]** 使用 `pinyin-data`_ v0.5.1 的拼音数据。(via `#125`_)


`0.30.0`_ (2018-02-03)
++++++++++++++++++++++++

* **[New]** 支持有拼音的非汉字字符 ``〇`` (via `#119`_)。
* **[Changed]** 修复之前无意中把 ``pinyin`` 函数中的 ``strict`` 参数的默认值修改为了 ``False`` ，
  现在把 ``strict`` 参数的默认值恢复为预期的 ``True`` (via `#121`_)。关于 ``strict`` 参数的影响详见文档： `strict 参数的影响`_


`0.29.0`_ (2018-01-14)
++++++++++++++++++++++++

* **[New]** 可以通过环境变量 ``PYPINYIN_NO_DICT_COPY`` 禁用代码内对 dict 的 copy 操作，节省内存(via `#115`_ thanks `@daya0576`_ )。

`0.28.0`_ (2017-12-08)
++++++++++++++++++++++++

* **[New]** 给代码增加类型注解(via `#110`_)。


`0.27.0`_ (2017-10-28)
++++++++++++++++++++++++

* **[New]** 命令行工具支持通过更简便的方式指定参数及拼音风格。
  (详见 `#105`_, Thanks `@wdscxsj`_ )
* **[Improved]** 增加说明 ``strict`` 参数对结果有什么影响的文档。


`0.26.1`_ (2017-10-25)
++++++++++++++++++++++++

* **[Improved]** 使用 `phrase-pinyin-data`_ v0.5.1 的词语拼音数据。fixed `#106`_


`0.26.0`_ (2017-10-12)
+++++++++++++++++++++++

* **[Changed]** 不再自动调用 jieba 分词模块，改为自动调用内置的正向最大匹配分词模块来分词。
  (via `#102`_)


`0.25.0`_ (2017-10-01)
+++++++++++++++++++++++

* **[New]** 内置一个正向最大匹配分词模块，使用内置的词语拼音库来训练这个分词模块，
  解决自定义词语库有时可能不生效的问题（因为这个词语在 jieba 等分词模块中不是可用词）。(via `#81`_)


  获取拼音或自定义词库后使用：

  .. code-block:: python

      >>> from pypinyin import pinyin, load_phrases_dict
      >>> load_phrases_dict({'了局': [['liǎo'], ['jú']]})
      >>> pinyin('了局啊')   # 使用 jieba 分词
      Building prefix dict from the default dictionary ...
      Dumping model to file cache /var/folders/s6/z9r_07h53pj_d4x7qjszwmbw0000gn/T/jieba.cache
      Loading model cost 1.175 seconds.
      Prefix dict has been built succesfully.
      [['le'], ['jú'], ['a']]

      >>> from pypinyin.contrib.mmseg import seg, retrain
      >>> retrain(seg)   # 没有使用 load_phrases_dict 时可以不调用这个函数
      >>> pinyin(seg.cut('了局啊'))  # 使用内置的正向最大匹配分词
      [['liǎo'], ['jú'], ['a']]
      >>>

  单独使用:

  .. code-block:: python

        >>> from pypinyin.contrib.mmseg import seg
        >>> text = '你好，我是中国人，我爱我的祖国'
        >>> seg.cut(text)
        <generator object Seg.cut at 0x10b2df2b0>
        >>> list(seg.cut(text))
        ['你好', '，', '我', '是', '中国人', '，', '我', '爱',
         '我的', '祖', '国']
        >>> seg.train(['祖国', '我是'])
        >>> list(seg.cut(text))
        ['你好', '，', '我是', '中国人', '，', '我', '爱',
         '我的', '祖国']
        >>>


`0.24.0`_ (2017-09-17)
++++++++++++++++++++++++

* **[New]** 支持类似 pyinstaller 的打包工具对使用 pypinyin 的程序进行打包，
  不会出现跟打包前不一样的输出（比如： `#92`_ ）（via `#93`_ ）。


`0.23.0`_ (2017-07-09)
++++++++++++++++++++++++

* **[New]** 使用 `phrase-pinyin-data`_ v0.5.0 的词语拼音数据。


`0.22.0`_ (2017-06-14)
++++++++++++++++++++++++

* **[New]** 支持 IronPython (via `#86`_). Thanks `@LevyLession`_


`0.21.1`_ (2017-05-29)
++++++++++++++++++++++++

* **[Bugfixed]** 修复在 Python 2 下通过 pip install 安装 wheel 格式的安装包后, 无法正常使用的问题。（Python 2 下没有自动安装依赖包）


`0.21.0`_ (2017-05-14)
++++++++++++++++++++++++

* **[New]** 重构各拼音风格实现，支持自定义拼音风格或覆盖已有拼音风格的实现.

  .. code-block:: python

      from pypinyin.style import register

      @register('style1')
      def func(pinyin, **kwargs):
          # pinyin = xxx   # convert to style1
          return pinyin

      def func(pinyin, **kwargs):
          # pinyin = xxx   # convert to style2
          return pinyin
      register('style2', func=func)


`0.20.0`_ (2017-05-13)
++++++++++++++++++++++++

* **[New]** 增加 ``strict`` 参数来控制处理声母和韵母时是否严格遵循 `《汉语拼音方案》 <http://www.moe.edu.cn/s78/A19/yxs_left/moe_810/s230/195802/t19580201_186000.html>`_ 标准。

  当 ``strict=True`` 时根据 `《汉语拼音方案》 <http://www.moe.edu.cn/s78/A19/yxs_left/moe_810/s230/195802/t19580201_186000.html>`_ 的如下规则处理声母、在韵母相关风格下还原正确的韵母：

   * 21 个声母： ``b p m f d t n l g k h j q x zh ch sh r z c s`` （**y, w 不是声母**）
   * i行的韵母，前面没有声母的时候，写成yi(衣)，ya(呀)，ye(耶)，yao(腰)，you(忧)，yan(烟)，yin(因)，yang(央)，ying(英)，yong(雍)。（**y 不是声母**）
   * u行的韵母，前面没有声母的时候，写成wu(乌)，wa(蛙)，wo(窝)，wai(歪)，wei(威)，wan(弯)，wen(温)，wang(汪)，weng(翁)。（**w 不是声母**）
   * ü行的韵母，前面没有声母的时候，写成yu(迂)，yue(约)，yuan(冤)，yun(晕)；ü上两点省略。（**韵母相关风格下还原正确的韵母 ü**）
   * ü行的韵跟声母j，q，x拼的时候，写成ju(居)，qu(区)，xu(虚)，ü上两点也省略；
     但是跟声母n，l拼的时候，仍然写成nü(女)，lü(吕)。（**韵母相关风格下还原正确的韵母 ü**）
   * iou，uei，uen前面加声母的时候，写成iu，ui，un。例如niu(牛)，gui(归)，lun(论)。（**韵母相关风格下还原正确的韵母 iou，uei，uen**）

  具体差异可以查看 tests/test_standard.py 中的对比结果测试用例

* **[Changed]** 改为使用 enum 定义拼音风格（兼容旧版本）


`0.19.0`_ (2017-05-05)
++++++++++++++++++++++++

* **[New]** 韵母风格下根据 `汉语拼音方案`_ 还原原始的 ``iou`` , ``uei`` , ``uen`` 韵母。

    iou，uei，uen前面加声母的时候，写成iu，ui，un。
    例如niu(牛)，gui(归)，lun(论)。即：

    * niu 的韵母是 iou
    * gui 的韵母是 uei
    * lun 的韵母是 uen
* **[Fixed]** 修复韵母相关风格下没有正确处理 ``wu`` 的韵母的问题
  (比如: ``无`` 在 ``FINALS_TONE`` 风格下的结果是 ``uú`` 的问题) 。
* **[Fixed]** 修复漏了 ǖ -> v1 的转换。



`0.18.2`_ (2017-04-25)
++++++++++++++++++++++++

* **[Fixed]** 使用 `phrase-pinyin-data`_ v0.4.1 的词语拼音数据, fixed `#72`_ 。


`0.18.1`_ (2017-03-22)
++++++++++++++++++++++++

* **[Improved]** PyPI 上传过程中出了点问题。


`0.18.0`_ (2017-03-22)
++++++++++++++++++++++++

* **[Changed]** 使用 `phrase-pinyin-data`_ v0.4.0 的词语拼音数据。


`0.17.0`_ (2017-03-13)
++++++++++++++++++++++++

* **[Changed]** 词语拼音数据改为使用来自 `phrase-pinyin-data`_ v0.3.1 的拼音数据。
* **[Fixed]** 修正 ``斯事体大`` 的拼音。


`0.16.1`_ (2017-02-12)
++++++++++++++++++++++++

* **[Improved]** 使用 `pinyin-data`_ v0.4.1 的拼音数据. fixed `#58`_
* **[Improved]** 更新 `厦门` 的拼音. fixed `#59`_


`0.16.0`_ (2016-11-27)
++++++++++++++++++++++++

* **[New]** Added new pinyin styles - ``CYRILLIC`` (汉语拼音与俄语字母对照表) and ``CYRILLIC _FIRST`` (via `#55`_ thanks `@tyrbonit`_)

  .. code-block:: python

      >>> pypinyin.pinyin('中心', style=pypinyin.CYRILLIC)
      [['чжун1'], ['синь1']]
      >>> pypinyin.pinyin('中心', style=pypinyin.CYRILLIC_FIRST)
      [['ч'], ['с']]
* **[New]** Added Russian translation README (`README_ru.rst`_)
* **[New]** Command-line tool supported the new pinyin styles: ``CYRILLIC, CYRILLIC_FIRST``


`0.15.0`_ (2016-10-18)
++++++++++++++++++++++++

* **[Changed]** 使用 `pinyin-data`_ v0.4.0 的拼音数据


`0.14.0`_ (2016-09-24)
++++++++++++++++++++++++

* **[New]** 新增注音 ``BOPOMOFO`` 及注音首字母 ``BOPOMOFO_FIRST`` 风格(via `#51`_ thanks `@gumblex`_ `@Artoria2e5`_)

  .. code-block:: python

      >>> pypinyin.pinyin('中心', style=pypinyin.BOPOMOFO)
      [['ㄓㄨㄥ'], ['ㄒㄧㄣ']]
      >>> pypinyin.pinyin('中心', style=pypinyin.BOPOMOFO_FIRST)
      [['ㄓ'], ['ㄒ']]


* **[New]** 新增音调在拼音后的 ``TONE3`` 以及 ``FINALS_TONE3`` 风格(via `#51`_ thanks `@gumblex`_ `@Artoria2e5`_ )

  .. code-block:: python

      >>> pypinyin.pinyin('中心', style=pypinyin.TONE3)
      [['zhong1'], ['xin1']]
      >>> pypinyin.pinyin('中心', style=pypinyin.FINALS_TONE3)
      [['ong1'], ['in1']]

* **[New]** 命令行程序支持新增的四个风格: ``TONE3, FINALS_TONE3, BOPOMOFO, BOPOMOFO_FIRST``
* **[Bugfixed]** 修复 TONE2 中 ü 标轻声的问题（像 侵略 -> qi1n lv0e4），以及去除文档中 0 表示轻声(via `#51`_ thanks `@gumblex`_)
* **[Changed]** 不再使用 0 表示轻声，轻声时没有数字(via `#51`_ thanks `@gumblex`_)


`0.13.0`_ (2016-08-19)
++++++++++++++++++++++++

* **[Changed]** 分离词组库中包含中文逗号的词语(via `f097b6a <https://github.com/mozillazg/python-pinyin/commit/f097b6ad7b9e2acbc1ecc214991be510f4f95d72>`_)
* **[Changed]** 使用 `pinyin-data`_ v0.3.0 的拼音数据


`0.12.1`_ (2016-05-11)
++++++++++++++++++++++++

* **[Bugfixed]** 修复一些词语存在拼音粘连在一起的情况. (`#41`_ thanks `@jolly-tao`_ )


`0.12.0`_ (2016-03-12)
++++++++++++++++++++++++

* **[Changed]** 单个汉字的拼音数据改为使用来自 `pinyin-data`_ 的拼音数据。
* **[New]** 命令行程序支持从标准输入读取汉字信息::

    $ echo "你好" | pypinyin
    nǐ hǎo
    $ pypinyin < hello.txt
    nǐ hǎo


`0.11.1`_ (2016-02-17)
+++++++++++++++++++++++

* **[Bugfixed]** 更新 phrases_dict 修复类似 `#36`_ 的问题。thanks `@someus`_


`0.11.0`_ (2016-01-16)
+++++++++++++++++++++++

* **[Changed]** 分割 ``__init__.py`` 为 ``compat.py``, ``constants.py``， ``core.py`` 和 ``utils.py``。
  影响: ``__init__.py`` 中只保留文档中提到过的 api, 如果使用了不在文档中的 api 则需要调整代码。


`0.10.0`_ (2016-01-02)
+++++++++++++++++++++++

* **[New]** Python 3.3++++ 以上版本默认支持 ``U++++20000 ~ U++++2FA1F`` 区间内的汉字(详见 `#33`_)


`0.9.5`_ (2015-12-19)
+++++++++++++++++++++++

* **[Bugfixed]** 修复未正确处理鼻音（详见 `汉语拼音 - 维基百科`_ ）的问题(`#31`_ thanks `@xulin97`_ ):

  * ``ḿ、ń、ň、ǹ`` 对应 “呒”、“呣”、“唔”、“嗯”等字。
    这些字之前在各种风格下都输出原始的汉字而不是拼音。


`0.9.4`_ (2015-11-27)
+++++++++++++++++++++++

* **[Improved]** 细微调整，主要是更新文档


`0.9.3`_ (2015-11-15)
+++++++++++++++++++++++

* **[Bugfixed]** Fixed Python 3 compatibility was broken.


`0.9.2`_ (2015-11-15)
+++++++++++++++++++++++

* **[New]** ``load_single_dict`` 和 ``load_phrases_dict`` 增加 ``style`` 参数支持 TONE2 风格的拼音 ::

      load_single_dict({ord(u'啊'): 'a1'}, style='tone2')
      load_phrases_dict({u"阿爸": [[u"a1"], [u"ba4"]]}, style='tone2'}
* **[Improved]** Improved docs


`0.9.1`_ (2015-10-17)
+++++++++++++++++++++++

* **[Bugfixed][Changed]** 修复 ``ju``, ``qu``, ``xu``, ``yu``, ``yi`` 和 ``wu`` 的韵母( `#26`_ ). Thanks `@MingStar`_ :

  * ``ju``, ``qu``, ``xu`` 的韵母应该是 ``v``
  * ``yi`` 的韵母是 ``i``
  * ``wu`` 的韵母是 ``u``
  * 从现在开始 ``y`` 既不是声母也不是韵母，详见 `汉语拼音方案`_


`0.9.0`_ (2015-09-20)
+++++++++++++++++++++++

* **[Changed]** 将拼音词典库里的国际音标字母替换为 ASCII 字母. Thanks `@MingStar`_ :

  * ``ɑ -> a``
  * ``ɡ -> g``


`0.8.5`_ (2015-08-23)
+++++++++++++++++++++++

* **[Bugfixed]** 修复 zh, ch, sh, z, c, s 顺序问题导致获取声母有误


`0.8.4`_ (2015-08-23)
+++++++++++++++++++++++

* **[Changed]** ``y``, ``w`` 也不是声母. (`hotoo/pinyin#57 <https://github.com/hotoo/pinyin/issues/57>`__):

  * 以 ``y``, ``w`` 开头的拼音在声母(``INITIALS``)模式下将返回 ``['']``


`0.8.3`_ (2015-08-20)
+++++++++++++++++++++++

* **[Improved]** 上传到 PyPI 出了点问题，但是又 `没法重新上传 <http://sourceforge.net/p/pypi/support-requests/468/>`__ ，只好新增一个版本


`0.8.2`_ (2015-08-20)
+++++++++++++++++++++++

* **[Bugfixed][Changed]** 修复误把 yu 放入声母列表里的 BUG(`#22`_). Thanks `@MingStar`_


`0.8.1`_ (2015-07-04)
+++++++++++++++++++++++

* **[Bugfixed]** 重构内置的分词功能，修复“无法正确处理包含空格的字符串的问题”


`0.8.0`_ (2015-06-27)
++++++++++++++++++++++++

* **[New]** 内置简单的分词功能，完善处理没有拼音的字符
  （如果不需要处理多音字问题, 现在可以不用安装 ``jieba`` 或其他分词模块了）::

        # 之前, 安装了结巴分词模块
        lazy_pinyin(u'你好abc☆☆')
        [u'ni', u'hao', 'a', 'b', 'c', u'\u2606', u'\u2606']

        # 现在, 无论是否安装结巴分词模块
        lazy_pinyin(u'你好abc☆☆')
        [u'ni', u'hao', u'abc\u2606\u2606']

* | **[Changed]** 当 ``errors`` 参数是回调函数时，函数的参数由 ``单个字符`` 变更为 ``单个字符或词组`` 。
  | 即: 对于 ``abc`` 字符串, 之前将调用三次 ``errors`` 回调函数: ``func('a') ... func('b') ... func('abc')``
  | 现在只调用一次: ``func('abc')`` 。
* **[Changed]** 将英文字符也纳入 ``errors`` 参数的处理范围::

        # 之前
        lazy_pinyin(u'abc', errors='ignore')
        [u'abc']

        # 现在
        lazy_pinyin(u'abc', errors='ignore')
        []

`0.7.0`_ (2015-06-20)
++++++++++++++++++++++++

* **[Bugfixed]** Python 2 下无法使用 ``from pypinyin import *`` 的问题
* **[New]** 支持以下环境变量:

  * ``PYPINYIN_NO_JIEBA=true``: 禁用“自动调用结巴分词模块”
  * ``PYPINYIN_NO_PHRASES=true``: 禁用内置的“词组拼音库”


`0.6.0`_ (2015-06-10)
++++++++++++++++++++++++

* **[New]** ``errors`` 参数支持回调函数(`#17`_): ::

    def foobar(char):
        return u'a'
    pinyin(u'あ', errors=foobar)

`0.5.7`_ (2015-05-17)
++++++++++++++++++++++

* **[Bugfixed]** 纠正包含 "便宜" 的一些词组的读音


`0.5.6`_ (2015-02-26)
++++++++++++++++++++++

* **[Bugfixed]** "苹果" pinyin error. `#11`__
* **[Bugfixed]** 重复 import jieba 的问题
* **[Improved]** 精简 phrases_dict
* **[Improved]** 更新文档

__ https://github.com/mozillazg/python-pinyin/issues/11


`0.5.5`_ (2015-01-27)
++++++++++++++++++++++

* **[Bugfixed]** phrases_dict error


`0.5.4`_ (2014-12-26)
++++++++++++++++++++++

* **[Bugfixed]** 无法正确处理由分词模块产生的中英文混合词组（比如：B超，维生素C）的问题.  `#8`__

__ https://github.com/mozillazg/python-pinyin/issues/8


`0.5.3`_ (2014-12-07)
++++++++++++++++++++++

* **[Improved]** 更新拼音库


`0.5.2`_ (2014-09-21)
+++++++++++++++++++++

* **[Improved]** 载入拼音库时，改为载入其副本。防止内置的拼音库被破坏
* **[Bugfixed]** ``胜败乃兵家常事`` 的音标问题


`0.5.1`_ (2014-03-09)
+++++++++++++++++++++

* **[New]** 参数 ``errors`` 用来控制如何处理没有拼音的字符:

  * ``'default'``: 保留原始字符
  * ``'ignore'``: 忽略该字符
  * ``'replace'``: 替换为去掉 ``\u`` 的 unicode 编码字符串(``u'\u90aa'`` => ``u'90aa'``)

  只处理 ``[^a-zA-Z0-9_]`` 字符。


`0.5.0`_ (2014-03-01)
+++++++++++++++++++++

* **[Changed]** **使用新的单字拼音库内容和格式**

  | 新的格式：``{0x963F: u"ā,ē"}``
  | 旧的格式：``{u'啊': u"ā,ē"}``


`0.4.4`_ (2014-01-16)
+++++++++++++++++++++

* **[Improved]** 清理命令行命令的输出结果，去除无关信息
* **[Bugfixed]** “ImportError: No module named runner”


`0.4.3`_ (2014-01-10)
+++++++++++++++++++++

* **[Bugfixed]** 命令行工具在 Python 3 下的兼容性问题


`0.4.2`_ (2014-01-10)
+++++++++++++++++++++

* **[Changed]** 拼音风格前的 ``STYLE_`` 前缀（兼容包含 ``STYLE_`` 前缀的拼音风格）
* **[New]** 命令行工具，具体用法请见： ``pypinyin -h``


`0.4.1`_ (2014-01-04)
+++++++++++++++++++++

* **[New]** 支持自定义拼音库，方便用户修正程序结果(``load_single_dict``, ``load_phrases_dict``)


`0.4.0`_ (2014-01-03)
+++++++++++++++++++++

* **[Changed]** 将 ``jieba`` 模块改为可选安装，用户可以选择使用自己喜爱的分词模块对汉字进行分词处理
* **[New]** 支持 Python 3


`0.3.1`_ (2013-12-24)
+++++++++++++++++++++

* **[New]** ``lazy_pinyin`` ::

    >>> lazy_pinyin(u'中心')
    ['zhong', 'xin']


`0.3.0`_ (2013-09-26)
+++++++++++++++++++++

* **[Bugfixed]** 首字母风格无法正确处理只有韵母的汉字

* **[New]** 三个拼音风格:
    * ``pypinyin.STYLE_FINALS`` ：       韵母风格1，只返回各个拼音的韵母部分，不带声调。如： ``ong uo``
    * ``pypinyin.STYLE_FINALS_TONE`` ：   韵母风格2，带声调，声调在韵母第一个字母上。如： ``ōng uó``
    * ``pypinyin.STYLE_FINALS_TONE2`` ：  韵母风格2，带声调，声调在各个拼音之后，用数字 [0-4] 进行表示。如： ``o1ng uo2``


`0.2.0`_ (2013-09-22)
+++++++++++++++++++++

* **[Improved]** 完善对中英文混合字符串的支持::

    >> pypinyin.pinyin(u'你好abc')
    [[u'n\u01d0'], [u'h\u01ceo'], [u'abc']]


0.1.0 (2013-09-21)
++++++++++++++++++

* **[New]** Initial Release


.. _#17: https://github.com/mozillazg/python-pinyin/pull/17
.. _#22: https://github.com/mozillazg/python-pinyin/pull/22
.. _#26: https://github.com/mozillazg/python-pinyin/pull/26
.. _@MingStar: https://github.com/MingStar
.. _汉语拼音方案: https://zh.wiktionary.org/wiki/%E9%99%84%E5%BD%95:%E6%B1%89%E8%AF%AD%E6%8B%BC%E9%9F%B3%E6%96%B9%E6%A1%88
.. _汉语拼音方案.pdf: http://www.moe.edu.cn/s78/A19/yxs_left/moe_810/s230/195802/t19580201_186000.html
.. _汉语拼音 - 维基百科: https://zh.wikipedia.org/wiki/%E6%B1%89%E8%AF%AD%E6%8B%BC%E9%9F%B3#cite_ref-10
.. _@xulin97: https://github.com/xulin97
.. _#31: https://github.com/mozillazg/python-pinyin/issues/31
.. _#33: https://github.com/mozillazg/python-pinyin/pull/33
.. _#36: https://github.com/mozillazg/python-pinyin/issues/36
.. _pinyin-data: https://github.com/mozillazg/pinyin-data
.. _@someus: https://github.com/someus
.. _#34: https://github.com/mozillazg/python-pinyin/issues/34
.. _#41: https://github.com/mozillazg/python-pinyin/issues/41
.. _@jolly-tao: https://github.com/jolly-tao
.. _@gumblex: https://github.com/gumblex
.. _@Artoria2e5: https://github.com/Artoria2e5
.. _#51: https://github.com/mozillazg/python-pinyin/issues/51
.. _#55: https://github.com/mozillazg/python-pinyin/pull/55
.. _@tyrbonit: https://github.com/tyrbonit
.. _README_ru.rst: https://github.com/mozillazg/python-pinyin/blob/master/README_ru.rst
.. _#58: https://github.com/mozillazg/python-pinyin/issues/58
.. _#59: https://github.com/mozillazg/python-pinyin/issues/59
.. _#72: https://github.com/mozillazg/python-pinyin/issues/72
.. _phrase-pinyin-data: https://github.com/mozillazg/phrase-pinyin-data
.. _@LevyLession: https://github.com/LevyLession
.. _#86: https://github.com/mozillazg/python-pinyin/issues/86
.. _#92: https://github.com/mozillazg/python-pinyin/issues/92
.. _#93: https://github.com/mozillazg/python-pinyin/issues/93
.. _#81: https://github.com/mozillazg/python-pinyin/issues/81
.. _#102: https://github.com/mozillazg/python-pinyin/issues/102
.. _#105: https://github.com/mozillazg/python-pinyin/issues/105
.. _#106: https://github.com/mozillazg/python-pinyin/issues/106
.. _@wdscxsj: https://github.com/wdscxsj
.. _#110: https://github.com/mozillazg/python-pinyin/pull/110
.. _#115: https://github.com/mozillazg/python-pinyin/pull/115
.. _#119: https://github.com/mozillazg/python-pinyin/pull/119
.. _@daya0576: https://github.com/daya0576
.. _#121: https://github.com/mozillazg/python-pinyin/pull/121
.. _#125: https://github.com/mozillazg/python-pinyin/pull/125
.. _#126: https://github.com/mozillazg/python-pinyin/pull/126
.. _#112: https://github.com/mozillazg/python-pinyin/issues/112
.. _#117: https://github.com/mozillazg/python-pinyin/issues/117
.. _#122: https://github.com/mozillazg/python-pinyin/issues/122
.. _#131: https://github.com/mozillazg/python-pinyin/issues/131
.. _#130: https://github.com/mozillazg/python-pinyin/pull/130
.. _PEP 561: https://www.python.org/dev/peps/pep-0561/
.. _#137: https://github.com/mozillazg/python-pinyin/issues/137
.. _#147: https://github.com/mozillazg/python-pinyin/pull/147
.. _@howl-anderson: https://github.com/howl-anderson
.. _#151: https://github.com/mozillazg/python-pinyin/issues/151
.. _#154: https://github.com/mozillazg/python-pinyin/issues/154
.. _#149: https://github.com/mozillazg/python-pinyin/issues/149
.. _#159: https://github.com/mozillazg/python-pinyin/issues/159
.. _#160: https://github.com/mozillazg/python-pinyin/issues/160
.. _strict 参数的影响: https://pypinyin.readthedocs.io/zh_CN/master/usage.html#strict
.. _#166: https://github.com/mozillazg/python-pinyin/issues/166
.. _#167: https://github.com/mozillazg/python-pinyin/issues/167
.. _#169: https://github.com/mozillazg/python-pinyin/issues/169
.. _#170: https://github.com/mozillazg/python-pinyin/issues/170
.. _#174: https://github.com/mozillazg/python-pinyin/issues/174
.. _#139: https://github.com/mozillazg/python-pinyin/issues/139
.. _#205: https://github.com/mozillazg/python-pinyin/issues/205
.. _#164: https://github.com/mozillazg/python-pinyin/pull/164
.. _#176: https://github.com/mozillazg/python-pinyin/pull/176
.. _@hanabi1224: https://github.com/hanabi1224
.. _@yangwe1: https://github.com/yangwe1


.. _0.2.0: https://github.com/mozillazg/python-pinyin/compare/v0.1.0...v0.2.0
.. _0.3.0: https://github.com/mozillazg/python-pinyin/compare/v0.2.0...v0.3.0
.. _0.3.1: https://github.com/mozillazg/python-pinyin/compare/v0.3.0...v0.3.1
.. _0.4.0: https://github.com/mozillazg/python-pinyin/compare/v0.3.1...v0.4.0
.. _0.4.1: https://github.com/mozillazg/python-pinyin/compare/v0.4.0...v0.4.1
.. _0.4.2: https://github.com/mozillazg/python-pinyin/compare/v0.4.1...v0.4.2
.. _0.4.3: https://github.com/mozillazg/python-pinyin/compare/v0.4.2...v0.4.3
.. _0.4.4: https://github.com/mozillazg/python-pinyin/compare/v0.4.3...v0.4.4
.. _0.5.0: https://github.com/mozillazg/python-pinyin/compare/v0.4.4...v0.5.0
.. _0.5.1: https://github.com/mozillazg/python-pinyin/compare/v0.5.0...v0.5.1
.. _0.5.2: https://github.com/mozillazg/python-pinyin/compare/v0.5.1...v0.5.2
.. _0.5.3: https://github.com/mozillazg/python-pinyin/compare/v0.5.2...v0.5.3
.. _0.5.4: https://github.com/mozillazg/python-pinyin/compare/v0.5.3...v0.5.4
.. _0.5.5: https://github.com/mozillazg/python-pinyin/compare/v0.5.4...v0.5.5
.. _0.5.6: https://github.com/mozillazg/python-pinyin/compare/v0.5.5...v0.5.6
.. _0.5.7: https://github.com/mozillazg/python-pinyin/compare/v0.5.6...v0.5.7
.. _0.6.0: https://github.com/mozillazg/python-pinyin/compare/v0.5.7...v0.6.0
.. _0.7.0: https://github.com/mozillazg/python-pinyin/compare/v0.6.0...v0.7.0
.. _0.8.0: https://github.com/mozillazg/python-pinyin/compare/v0.7.0...v0.8.0
.. _0.8.1: https://github.com/mozillazg/python-pinyin/compare/v0.8.0...v0.8.1
.. _0.8.2: https://github.com/mozillazg/python-pinyin/compare/v0.8.1...v0.8.2
.. _0.8.3: https://github.com/mozillazg/python-pinyin/compare/v0.8.2...v0.8.3
.. _0.8.4: https://github.com/mozillazg/python-pinyin/compare/v0.8.3...v0.8.4
.. _0.8.5: https://github.com/mozillazg/python-pinyin/compare/v0.8.4...v0.8.5
.. _0.9.0: https://github.com/mozillazg/python-pinyin/compare/v0.8.5...v0.9.0
.. _0.9.1: https://github.com/mozillazg/python-pinyin/compare/v0.9.0...v0.9.1
.. _0.9.2: https://github.com/mozillazg/python-pinyin/compare/v0.9.1...v0.9.2
.. _0.9.3: https://github.com/mozillazg/python-pinyin/compare/v0.9.2...v0.9.3
.. _0.9.4: https://github.com/mozillazg/python-pinyin/compare/v0.9.3...v0.9.4
.. _0.9.5: https://github.com/mozillazg/python-pinyin/compare/v0.9.4...v0.9.5
.. _0.10.0: https://github.com/mozillazg/python-pinyin/compare/v0.9.5...v0.10.0
.. _0.11.0: https://github.com/mozillazg/python-pinyin/compare/v0.10.0...v0.11.0
.. _0.11.1: https://github.com/mozillazg/python-pinyin/compare/v0.11.0...v0.11.1
.. _0.12.0: https://github.com/mozillazg/python-pinyin/compare/v0.11.1...v0.12.0
.. _0.12.1: https://github.com/mozillazg/python-pinyin/compare/v0.12.0...v0.12.1
.. _0.13.0: https://github.com/mozillazg/python-pinyin/compare/v0.12.1...v0.13.0
.. _0.14.0: https://github.com/mozillazg/python-pinyin/compare/v0.13.0...v0.14.0
.. _0.15.0: https://github.com/mozillazg/python-pinyin/compare/v0.14.0...v0.15.0
.. _0.16.0: https://github.com/mozillazg/python-pinyin/compare/v0.15.0...v0.16.0
.. _0.16.1: https://github.com/mozillazg/python-pinyin/compare/v0.16.0...v0.16.1
.. _0.17.0: https://github.com/mozillazg/python-pinyin/compare/v0.16.1...v0.17.0
.. _0.18.0: https://github.com/mozillazg/python-pinyin/compare/v0.17.0...v0.18.0
.. _0.18.1: https://github.com/mozillazg/python-pinyin/compare/v0.18.0...v0.18.1
.. _0.18.2: https://github.com/mozillazg/python-pinyin/compare/v0.18.1...v0.18.2
.. _0.19.0: https://github.com/mozillazg/python-pinyin/compare/v0.18.2...v0.19.0
.. _0.20.0: https://github.com/mozillazg/python-pinyin/compare/v0.19.0...v0.20.0
.. _0.21.0: https://github.com/mozillazg/python-pinyin/compare/v0.20.0...v0.21.0
.. _0.21.1: https://github.com/mozillazg/python-pinyin/compare/v0.21.0...v0.21.1
.. _0.22.0: https://github.com/mozillazg/python-pinyin/compare/v0.21.1...v0.22.0
.. _0.23.0: https://github.com/mozillazg/python-pinyin/compare/v0.22.0...v0.23.0
.. _0.24.0: https://github.com/mozillazg/python-pinyin/compare/v0.23.0...v0.24.0
.. _0.25.0: https://github.com/mozillazg/python-pinyin/compare/v0.24.0...v0.25.0
.. _0.26.0: https://github.com/mozillazg/python-pinyin/compare/v0.25.0...v0.26.0
.. _0.26.1: https://github.com/mozillazg/python-pinyin/compare/v0.26.0...v0.26.1
.. _0.27.0: https://github.com/mozillazg/python-pinyin/compare/v0.26.1...v0.27.0
.. _0.28.0: https://github.com/mozillazg/python-pinyin/compare/v0.27.0...v0.28.0
.. _0.29.0: https://github.com/mozillazg/python-pinyin/compare/v0.28.0...v0.29.0
.. _0.30.0: https://github.com/mozillazg/python-pinyin/compare/v0.29.0...v0.30.0
.. _0.30.1: https://github.com/mozillazg/python-pinyin/compare/v0.30.0...v0.30.1
.. _0.31.0: https://github.com/mozillazg/python-pinyin/compare/v0.30.1...v0.31.0
.. _0.32.0: https://github.com/mozillazg/python-pinyin/compare/v0.31.0...v0.32.0
.. _0.33.0: https://github.com/mozillazg/python-pinyin/compare/v0.32.0...v0.33.0
.. _0.33.1: https://github.com/mozillazg/python-pinyin/compare/v0.33.0...v0.33.1
.. _0.33.2: https://github.com/mozillazg/python-pinyin/compare/v0.33.1...v0.33.2
.. _0.34.0: https://github.com/mozillazg/python-pinyin/compare/v0.33.2...v0.34.0
.. _0.34.1: https://github.com/mozillazg/python-pinyin/compare/v0.34.0...v0.34.1
.. _0.35.0: https://github.com/mozillazg/python-pinyin/compare/v0.34.1...v0.35.0
.. _0.35.1: https://github.com/mozillazg/python-pinyin/compare/v0.35.0...v0.35.1
.. _0.35.2: https://github.com/mozillazg/python-pinyin/compare/v0.35.1...v0.35.2
.. _0.35.3: https://github.com/mozillazg/python-pinyin/compare/v0.35.2...v0.35.3
.. _0.35.4: https://github.com/mozillazg/python-pinyin/compare/v0.35.3...v0.35.4
.. _0.36.0: https://github.com/mozillazg/python-pinyin/compare/v0.35.4...v0.36.0
.. _0.37.0: https://github.com/mozillazg/python-pinyin/compare/v0.36.0...v0.37.0
.. _0.38.0: https://github.com/mozillazg/python-pinyin/compare/v0.37.0...v0.38.0
.. _0.38.1: https://github.com/mozillazg/python-pinyin/compare/v0.38.0...v0.38.1
.. _0.39.0: https://github.com/mozillazg/python-pinyin/compare/v0.38.1...v0.39.0
.. _0.39.1: https://github.com/mozillazg/python-pinyin/compare/v0.39.0...v0.39.1
.. _0.40.0: https://github.com/mozillazg/python-pinyin/compare/v0.39.1...v0.40.0
.. _0.41.0: https://github.com/mozillazg/python-pinyin/compare/v0.40.0...v0.41.0
