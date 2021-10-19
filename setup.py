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
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('parakeet', '__init__.py')
long_description = read("README.md")

setup_info = dict(
    # Metadata
    name='paddle-parakeet',
    version=VERSION,
    author='PaddleSL Team',
    author_email='',
    url='https://github.com/PaddlePaddle',
    description='Speech synthesis tools and models based on Paddlepaddle',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache 2',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'nltk',
        'inflect',
        'librosa',
        'unidecode',
        'numba',
        'tqdm',
        'llvmlite',
        'matplotlib',
        'visualdl==2.2.0',
        'scipy',
        'pandas',
        'sox',
        'soundfile~=0.10',
        'g2p_en',
        'yacs',
        'pypinyin',
        'webrtcvad',
        'g2pM',
        'praatio~=4.1',
        "h5py",
        "timer",
        'jsonlines',
        'pyworld',
        'typeguard',
        'jieba',
        "phkit",
    ],
    extras_require={
        'doc': ["sphinx", "sphinx-rtd-theme", "numpydoc"],
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
    ], )

setup(**setup_info)
