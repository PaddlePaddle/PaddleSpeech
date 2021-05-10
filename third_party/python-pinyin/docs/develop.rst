.. _develop:

开发文档
========


准备开发环境
-------------

::

    $ virtualenv venv
    $ . venv/bin/activate
    (venv) $ pip install -U -r requirements_dev.txt
    (venv) $ pip install -e .
    (venv) $ pre-commit install


TODO: 把这个步骤放到一个 make 命令中。

.. note::

    推荐在 Python 3.6+ 环境下进行开发。


测试
------

可以通过 ``make test`` 命令在当前 Python 版本下运行单元测试: ::

    (venv) $ make test


可以通过 ``tox`` 测试程序在多个 Python 版本下的单元测试结果（这一步也可以在提 PR 的时候通过 CI 来运行）: ::

    (venv) $ tox


.. note::

    如果对测试有疑问或者有些测试实在无法通过，可以先提交 PR 大家一起来看看。


目录结构
--------

关键文件和目录 ::

    $ tree -L 2
    .
    ├── CHANGELOG.rst        # 更新日志
    ├── Makefile
    ├── README.rst
    ├── docs                 # 文档
    ├── gen_phrases_dict.py  # 生成 phrases_dict.py 的脚本
    ├── gen_pinyin_dict.py   # 生成 pinyin_dict.py 的脚本
    ├── phrase-pinyin-data   # gen_phrases_dict.py 使用的数据源
    ├── pinyin-data          # gen_pinyin_dict.py 使用的数据源
    ├── pypinyin             # pypinyin 模块源代码
    │   ├── __init__.py
    │   ├── __main__.py      # 命令行程序的入口
    │   ├── compat.py
    │   ├── constants.py
    │   ├── contrib          # 目前包含了一个分词模块
    │   ├── core.py          # pypinyin 模块的核心逻辑
    │   ├── phonetic_symbol.py
    │   ├── phrases_dict.py   # 词组的拼音数据，由 gen_phrases_dict.py 生成
    │   ├── pinyin_dict.py    # 单个汉字的拼音数据，由 gen_pinyin_dict.py 生成
    │   ├── runner.py         # 命令行程序的主逻辑
    │   ├── standard.py       # strict=True 时的拼音转换逻辑
    │   ├── style             # 各种拼音风格在 style 目录下实现
    │   ├── utils.py
    ├── pytest.ini
    ├── requirements_dev.txt
    ├── setup.cfg
    ├── setup.py
    ├── tests
    ├── tox.ini


实现思路/主逻辑
----------------

主逻辑:

1. 对输入的字符串按是否是汉字进行分词（``seg``）
2. 对分词结果的每个词条进行获取词条拼音的逻辑

   1. 检查词条是否是汉字，不是汉字则走处理没有拼音数据的逻辑（``handle_nopinyin``）
   2. 检查词条是否在 ``PHRASES_DICT`` 中，如果在直接取 ``PHRASES_DICT`` 中这个词条的拼音数据
   3. 如果词条不在 ``PHRASES_DICT`` 中，遍历词条包含的字符，每个字符进行 ``single_pinyin`` 逻辑处理
3. ``single_pinyin`` 的逻辑：

   1. 检查字符是否在 ``PINYIN_DICT`` 中，如果在的话，取 ``PINYIN_DICT`` 中这个字符的拼音数据
   2. 如果不在的话，走 ``handle_nopinyin`` 逻辑
4. ``handle_nopinyin`` 逻辑: 根据 ``errors`` 参数的值返回不同的结果。
5. 对上面的步骤获得的拼音数据按指定的拼音风格进行转换。


* ``PHRASES_DICT``：词组拼音数据
* ``PINYIN_DICT``: 单个汉字的拼音数据


TODO: 画流程图


发布新版本
----------

1. 切分到 develop 分支
2. rebase master 分支的代码: ``make rebase_master``
3. 通过 ``make gen_data`` 生成最新的数据文件
4. 通过 ``make test`` 跑测试
5. 更新 CHANGELOG
6. 提交代码
7. 检查 develop 分支的 CI 结果
8. 切换到 master 分支
9. 合并 develop 分支代码: ``make merge_dev``
10. 更新版本号:

    * 大改动(1.1.x -> 1.2.x)：``make bump_minor``
    * 小改动(1.1.1 -> 1.1.2)：``make bump_patch``
11. 发布到 test pypi: ``make publish_test``
12. 安装和测试发布到 test pypi 上的版本
13. 发布到 pypi: ``make publish``
14. 安装和测试发布到 pypi 上的版本
15. 提交 master 分支代码，更新 develop 分支代码，进入下一个开发阶段：``make start_next``
