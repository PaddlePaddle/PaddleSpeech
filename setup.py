# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


VERSION = '2.1.2'
long_description = read("README.md")
deps = [d.strip() for d in read('requirements.txt').split()]

setup_info = dict(
    # Metadata
    name='paddle-speech',
    version=VERSION,
    author='PaddleSL Speech Team',
    author_email='',
    url='https://github.com/PaddlePaddle/DeepSpeech',
    description='Speech tools and models based on Paddlepaddle',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache 2',
    python_requires='>=3.6',
    install_requires=deps,
    extras_require={
        'doc': [
            "sphinx", "sphinx-rtd-theme", "numpydoc", "myst_parser",
            "recommonmark>=0.5.0", "sphinx-markdown-tables", "sphinx-autobuild"
        ],
    },

    # Package info
    packages=find_packages(exclude=('tests', 'tests.*')),
    zip_safe=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)

setup(**setup_info)
