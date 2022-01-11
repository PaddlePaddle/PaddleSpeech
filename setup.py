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
import contextlib
import inspect
import io
import os
import subprocess as sp
import sys
from pathlib import Path

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

HERE = Path(os.path.abspath(os.path.dirname(__file__)))

requirements = {
    "install": [
        "editdistance",
        "g2p_en",
        "g2pM",
        "h5py",
        "inflect",
        "jieba",
        "jsonlines",
        "kaldiio",
        "librosa",
        "loguru",
        "matplotlib",
        "nara_wpe",
        "nltk",
        "pandas",
        "paddleaudio",
        "paddlenlp",
        "paddlespeech_feat",
        "praatio==5.0.0",
        "pypinyin",
        "python-dateutil",
        "pyworld",
        "resampy==0.2.2",
        "sacrebleu",
        "scipy",
        "sentencepiece~=0.1.96",
        "soundfile~=0.10",
        "textgrid",
        "timer",
        "tqdm",
        "typeguard",
        "visualdl",
        "webrtcvad",
        "yacs~=0.1.8",
    ],
    "develop": [
        "ConfigArgParse",
        "coverage",
        "gpustat",
        "paddlespeech_ctcdecoders",
        "phkit",
        "Pillow",
        "pybind11",
        "pypi-kenlm",
        "snakeviz",
        "sox",
        "soxbindings",
        "unidecode",
        "yq",
        "pre-commit",
    ]
}


@contextlib.contextmanager
def pushd(new_dir):
    old_dir = os.getcwd()
    os.chdir(new_dir)
    print(new_dir)
    yield
    os.chdir(old_dir)
    print(old_dir)


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def check_call(cmd: str, shell=False, executable=None):
    try:
        sp.check_call(
            cmd.split(),
            shell=shell,
            executable="/bin/bash" if shell else executable)
    except sp.CalledProcessError as e:
        print(
            f"{__file__}:{inspect.currentframe().f_lineno}: CMD: {cmd}, Error:",
            e.output,
            file=sys.stderr)
        raise e


def _remove(files: str):
    for f in files:
        f.unlink()


def _post_install(install_lib_dir):
    # tools/make
    tool_dir = HERE / "tools"
    _remove(tool_dir.glob("*.done"))
    with pushd(tool_dir):
        check_call("make")
    print("tools install.")

    # ctcdecoder
    ctcdecoder_dir = HERE / 'paddlespeech/s2t/decoders/ctcdecoder/swig'
    with pushd(ctcdecoder_dir):
        check_call("bash -e setup.sh")
    print("ctcdecoder install.")


class DevelopCommand(develop):
    def run(self):
        develop.run(self)
        # must after develop.run, or pkg install by shell will not see
        self.execute(_post_install, (self.install_lib, ), msg="Post Install...")


class InstallCommand(install):
    def run(self):
        install.run(self)


    # cmd: python setup.py upload
class UploadCommand(Command):
    description = "Build and publish the package."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            print("Removing previous dist/ ...")
            shutil.rmtree(str(HERE / "dist"))
        except OSError:
            pass
        print("Building source distribution...")
        sp.check_call([sys.executable, "setup.py", "sdist"])
        print("Uploading package to PyPi...")
        sp.check_call(["twine", "upload", "dist/*"])
        sys.exit()


setup_info = dict(
    # Metadata
    name='paddlespeech',
    version='0.1.0',
    author='PaddlePaddle Speech and Language Team',
    author_email='paddlesl@baidu.com',
    url='https://github.com/PaddlePaddle/PaddleSpeech',
    license='Apache 2.0',
    description='Speech tools and models based on Paddlepaddle',
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "speech",
        "asr",
        "tts",
        "speaker verfication",
        "speech classfication",
        "text frontend",
        "MFA",
        "paddlepaddle",
        "beam search",
        "ctcdecoder",
        "deepspeech2",
        "transformer",
        "conformer",
        "fastspeech",
        "vocoder",
        "pwgan",
        "gan",
    ],
    python_requires='>=3.7',
    install_requires=requirements["install"],
    extras_require={
        'develop':
        requirements["develop"],
        'doc': [
            "sphinx", "sphinx-rtd-theme", "numpydoc", "myst_parser",
            "recommonmark>=0.5.0", "sphinx-markdown-tables", "sphinx-autobuild"
        ],
    },
    cmdclass={
        'develop': DevelopCommand,
        'install': InstallCommand,
        'upload': UploadCommand,
    },

    # Package info
    packages=find_packages(include=('paddlespeech*')),
    zip_safe=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points={
        'console_scripts': ['paddlespeech=paddlespeech.cli.entry:_execute']
    })

setup(**setup_info)
