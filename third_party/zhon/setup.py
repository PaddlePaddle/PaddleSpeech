#!/usr/bin/env python3

import os
import sys


enc_open = open

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

with enc_open('README.rst', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='zhon',
    version='1.1.5',
    author='Thomas Roten',
    author_email='thomas@roten.us',
    url='https://github.com/tsroten/zhon',
    description=('Zhon provides constants used in Chinese text processing.'),
    long_description=long_description,
    packages=['zhon', 'zhon.cedict'],
    keywords=('chinese mandarin segmentation tokenization punctuation hanzi '
              'unicode radicals han cjk cedict cc-cedict traditional '
              'simplified characters pinyin zhuyin'),
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Linguistic',
    ],
    platforms='Any',
    test_suite='tests',
)
