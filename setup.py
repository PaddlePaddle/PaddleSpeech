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
from setuptools.command.test import test

HERE = Path(os.path.abspath(os.path.dirname(__file__)))

VERSION = '0.0.0'
COMMITID = 'none'

base = [
    "braceexpand",
    "editdistance",
    "g2p_en",
    "g2pM",
    "h5py",
    "hyperpyyaml",
    "inflect",
    "jsonlines",
    # paddleaudio align with librosa==0.8.1, which need numpy==1.23.x
    "numpy==1.23.5",
    "librosa==0.8.1",
    "scipy>=1.4.0, <=1.12.0",
    "loguru",
    "matplotlib<=3.8.4",
    "nara_wpe",
    "onnxruntime>=1.11.0",
    "opencc==1.1.6",
    "opencc-python-reimplemented",
    "pandas",
    "paddleaudio>=1.1.0",
    "paddlenlp>=2.4.8",
    "paddlepaddle-gpu==2.5.1",
    "paddleslim>=2.3.4",
    "ppdiffusers>=0.9.0",
    "paddlespeech_feat",
    "praatio>=5.0.0, <=5.1.1",
    "prettytable",
    "pypinyin<=0.44.0",
    "pypinyin-dict",
    "python-dateutil",
    "pyworld>=0.2.12",
    "pyyaml",
    "resampy",
    "sacrebleu",
    "textgrid",
    "timer",
    "ToJyutping==0.2.1",
    "typeguard==2.13.3",
    "webrtcvad",
    "yacs~=0.1.8",
    "zhon",
]

server = ["pattern_singleton", "websockets"]

requirements = {
    "install":
    base + server,
    "develop": [
        "ConfigArgParse",
        "coverage",
        "gpustat",
        "paddlespeech_ctcdecoders",
        "phkit",
        "pypi-kenlm",
        "snakeviz",
        "sox",
        "soxbindings",
        "unidecode",
        "yq",
        "pre-commit",
    ]
}


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


def check_output(cmd: str, shell=False):
    try:
        out_bytes = sp.check_output(cmd.split())
    except sp.CalledProcessError as e:
        out_bytes = e.output  # Output generated before error
        code = e.returncode  # Return code
        print(
            f"{__file__}:{inspect.currentframe().f_lineno}: CMD: {cmd}, Error:",
            out_bytes,
            file=sys.stderr)
    return out_bytes.strip().decode('utf8')


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


def _remove(files: str):
    for f in files:
        f.unlink()


################################# Install ##################################


def _post_install(install_lib_dir):
    # tools/make
    tool_dir = HERE / "tools"
    _remove(tool_dir.glob("*.done"))
    with pushd(tool_dir):
        check_call("make")
    print("tools install.")

    # ctcdecoder
    ctcdecoder_dir = HERE / 'third_party/ctc_decoders'
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


class TestCommand(test):
    def finalize_options(self):
        test.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # Run nose ensuring that argv simulates running nosetests directly
        import nose
        nose.run_exit(argv=['nosetests', '-w', 'tests'])


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


################################# Version ##################################
def write_version_py(filename='paddlespeech/__init__.py'):
    import paddlespeech
    if hasattr(paddlespeech,
               "__version__") and paddlespeech.__version__ == VERSION:
        return
    with open(filename, "a") as f:
        out_str = f"\n__version__ = '{VERSION}'\n"
        print(out_str)
        f.write(f"\n__version__ = '{VERSION}'\n")

    COMMITID = check_output("git rev-parse HEAD")
    with open(filename, 'a') as f:
        out_str = f"\n__commit__ = '{COMMITID}'\n"
        print(out_str)
        f.write(f"\n__commit__ = '{COMMITID}'\n")

    print(f"{inspect.currentframe().f_code.co_name} done")


def remove_version_py(filename='paddlespeech/__init__.py'):
    with open(filename, "r") as f:
        lines = f.readlines()
    with open(filename, "w") as f:
        for line in lines:
            if "__version__" in line or "__commit__" in line:
                continue
            f.write(line)
    print(f"{inspect.currentframe().f_code.co_name} done")


@contextlib.contextmanager
def version_info():
    write_version_py()
    yield
    remove_version_py()


################################# Steup ##################################
setup_info = dict(
    # Metadata
    name='paddlespeech',
    version=VERSION,
    author='PaddlePaddle Speech and Language Team',
    author_email='paddlesl@baidu.com',
    url='https://github.com/PaddlePaddle/PaddleSpeech',
    license='Apache 2.0',
    description='Speech tools and models based on Paddlepaddle',
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "SSL"
        "speech",
        "asr",
        "tts",
        "speaker verfication",
        "speech classfication",
        "text frontend",
        "MFA",
        "paddlepaddle",
        "paddleaudio",
        "streaming asr",
        "streaming tts",
        "beam search",
        "ctcdecoder",
        "deepspeech2",
        "wav2vec2",
        "hubert",
        "wavlm",
        "transformer",
        "conformer",
        "fastspeech2",
        "hifigan",
        "gan vocoders",
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
        'test': ['nose', 'torchaudio==0.10.2'],
    },
    cmdclass={
        'develop': DevelopCommand,
        'install': InstallCommand,
        'upload': UploadCommand,
        'test': TestCommand,
    },

    # Package info
    packages=find_packages(
        include=['paddlespeech*'], exclude=['utils', 'third_party']),
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
        'console_scripts': [
            'paddlespeech=paddlespeech.cli.entry:_execute',
            'paddlespeech_server=paddlespeech.server.entry:server_execute',
            'paddlespeech_client=paddlespeech.server.entry:client_execute'
        ]
    })

with version_info():
    setup(**setup_info, include_package_data=True)
