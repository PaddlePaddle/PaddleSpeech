# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import setuptools

# set the version here
VERSION = '0.2.0'


def write_version_py(filename='paddleaudio/__init__.py'):
    import paddleaudio
    if hasattr(paddleaudio,
               "__version__") and paddleaudio.__version__ == VERSION:
        return
    with open(filename, "a") as f:
        f.write(f"\n__version__ = '{VERSION}'\n")


def remove_version_py(filename='paddleaudio/__init__.py'):
    with open(filename, "r") as f:
        lines = f.readlines()
    with open(filename, "w") as f:
        for line in lines:
            if "__version__" not in line:
                f.write(line)


write_version_py()

setuptools.setup(
    name="paddleaudio",
    version=VERSION,
    author="",
    author_email="",
    description="PaddleAudio, in development",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(include=['paddleaudio*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy >= 1.15.0',
        'scipy >= 1.0.0',
        'resampy >= 0.2.2',
        'soundfile >= 0.9.0',
        'colorlog',
    ], )

remove_version_py()
