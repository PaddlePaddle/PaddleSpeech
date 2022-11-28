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
from typing import List
from typing import Tuple
from typing import Union

import distutils.command.clean
from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.test import test

from tools import setup_helpers

ROOT_DIR = Path(__file__).parent.resolve()

VERSION = '0.0.0'
COMMITID = 'none'

base = [
    "editdistance",
    "g2p_en",
    "g2pM",
    "h5py",
    "inflect",
    "jieba",
    "jsonlines",
    "kaldiio",
    "librosa==0.8.1",
    "loguru",
    "matplotlib",
    "nara_wpe",
    "onnxruntime==1.10.0",
    "opencc",
    "pandas",
    "paddlenlp",
    "paddlespeech_feat",
    "Pillow>=9.0.0",
    "praatio==5.0.0",
    "protobuf>=3.1.0, <=3.20.0",
    "pypinyin<=0.44.0",
    "pypinyin-dict",
    "python-dateutil",
    "pyworld==0.2.12",
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
    "prettytable",
    "zhon",
    "colorlog",
    "pathos == 0.2.8",
    "braceexpand",
    "pyyaml",
    "pybind11",
    "paddlelite",
    "paddleslim==2.3.4",
]

server = ["fastapi", "uvicorn", "pattern_singleton", "websockets"]

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


def check_output(cmd: Union[str, List[str], Tuple[str]], shell=False):
    try:

        if isinstance(cmd, (list, tuple)):
            cmds = cmd
        else:
            cmds = cmd.split()
        out_bytes = sp.check_output(cmds)

    except sp.CalledProcessError as e:
        out_bytes = e.output  # Output generated before error
        code = e.returncode  # Return code
        print(
            f"{__file__}:{inspect.currentframe().f_lineno}: CMD: {cmd}, Error:",
            out_bytes,
            file=sys.stderr)
    return out_bytes.strip().decode('utf8')


def _run_cmd(cmd):
    try:
        return subprocess.check_output(
            cmd, cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL).decode("ascii").strip()
    except Exception:
        return None


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
    tool_dir = ROOT_DIR / "tools"
    _remove(tool_dir.glob("*.done"))
    with pushd(tool_dir):
        check_call("make")
    print("tools install.")

    # ctcdecoder
    ctcdecoder_dir = ROOT_DIR / 'third_party/ctc_decoders'
    with pushd(ctcdecoder_dir):
        check_call("bash -e setup.sh")
    print("ctcdecoder install.")


class DevelopCommand(develop):
    def run(self):
        develop.run(self)
        # must after develop.run, or pkg install by shell will not see
        self.execute(_post_install, (self.install_lib, ), msg="Post Install...")


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
            shutil.rmtree(str(ROOT_DIR / "dist"))
        except OSError:
            pass
        print("Building source distribution...")
        sp.check_call([sys.executable, "setup.py", "sdist"])
        print("Uploading package to PyPi...")
        sp.check_call(["twine", "upload", "dist/*"])
        sys.exit()


################################# Version ##################################
def _get_version(sha):
    version = VERSION
    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha is not None:
        version += "+" + sha[:7]
    return version


def _make_version_file(version, sha):
    sha = "Unknown" if sha is None else sha
    version_path = ROOT_DIR / "paddlespeech" / "version.py"
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"__commit__ = '{sha}'\n")


################################# Steup ##################################
class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torchaudio extension
        for path in (ROOT_DIR / "paddlespeech").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


def main():

    sha = _run_cmd(["git", "rev-parse", "HEAD"])  # commit id
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])
    print("-- Git branch:", branch)
    print("-- Git SHA:", sha)
    print("-- Git tag:", tag)
    version = _get_version(sha)
    print("-- Building version", version)
    _make_version_file(version, sha)

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
            "speech",
            "asr",
            "tts",
            "streaming asr"
            "streaming tts"
            "audio process"
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
            "melgan",
            "mb-melgan",
            "hifigan",
            "gan",
            "wfst decoder",
        ],
        python_requires='>=3.7',
        install_requires=requirements["install"],
        extras_require={
            'develop':
            requirements["develop"],
            'doc': [
                "sphinx", "sphinx-rtd-theme", "numpydoc", "myst_parser",
                "recommonmark>=0.5.0", "sphinx-markdown-tables",
                "sphinx-autobuild"
            ],
            'test': ['nose', 'torchaudio==0.10.2'],
        },
        cmdclass={
            "build_ext": setup_helpers.CMakeBuild,
            'develop': DevelopCommand,
            'test': TestCommand,
            'upload': UploadCommand,
            "clean": clean,
        },

        # Package info
        packages=find_packages(include=('paddlespeech*')),
        ext_modules=setup_helpers.get_ext_modules(),
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

    setup(**setup_info)


if __name__ == '__main__':
    main()
