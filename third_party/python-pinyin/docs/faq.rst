FAQ
-----


.. _no_phrases:

如何禁用内置的“词组拼音库”
++++++++++++++++++++++++++++++++

设置环境变量 ``PYPINYIN_NO_PHRASES=true`` 即可


.. _no_dict_copy:

如何禁用默认的“拼音库”copy 操作
+++++++++++++++++++++++++++++++++++++++++++

设置环境变量 ``PYPINYIN_NO_DICT_COPY=true`` 即可.

副作用: 用户的自定义拼音库出现问题时, 无法回退到自带的拼音库.


.. _limit_memory:

如何减少内存占用
+++++++++++++++++++++

如果对拼音正确性不在意的话，可以按照上面所说的设置环境变量 ``PYPINYIN_NO_PHRASES``
和 ``PYPINYIN_NO_DICT_COPY`` 详见 `#13`_


.. _initials_problem:

``INITIALS`` 声母风格下，以 ``y``, ``w``, ``yu`` 开头的汉字返回空字符串
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

比如：

  .. code:: python

      pinyin('火影忍者', style=Style.INITIALS)
      [['h'], [''], ['r'], ['zh']]


因为 ``y``, ``w``, ``yu`` 都不是声母。参考:
`hotoo/pinyin#57 <https://github.com/hotoo/pinyin/issues/57>`__,
`#22 <https://github.com/mozillazg/python-pinyin/pull/22>`__,
`#27 <https://github.com/mozillazg/python-pinyin/issues/27>`__,
`#44 <https://github.com/mozillazg/python-pinyin/issues/44>`__

  声母风格（INITIALS）下，“雨”、“我”、“圆”等汉字返回空字符串，因为根据
  `《汉语拼音方案》 <http://www.moe.edu.cn/s78/A19/yxs_left/moe_810/s230/195802/t19580201_186000.html>`__ ，
  y，w，ü (yu) 都不是声母，在某些特定韵母无声母时，才加上 y 或 w，而 ü 也有其特定规则。
  如果你觉得这个给你带来了麻烦，那么也请小心一些无声母的汉字（如“啊”、“饿”、“按”、“昂”等）。
  这时候你也许需要的是首字母风格（FIRST_LETTER）。    —— @hotoo

如果觉得这个行为不是你想要的，就是想把 y 当成声母的话，可以指定 ``strict=False`` ， 这个可能会符合你的预期。详见 `strict 参数的影响`_


.. _#13: https://github.com/mozillazg/python-pinyin/issues/113
.. _strict 参数的影响: https://pypinyin.readthedocs.io/zh_CN/master/usage.html#strict
