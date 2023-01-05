# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import platform
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

VERSION = '1.1.0'
COMMITID = 'none'

base = [
    "kaldiio",
    "librosa==0.8.1",
    "scipy>=1.0.0",
    "soundfile~=0.10",
    "colorlog",
    "pathos == 0.2.8",
    "pybind11",
    "parameterized",
    "tqdm",
    "scikit-learn"
]

requirements = {
    "install": base,
    "develop": [
        "sox",
        "soxbindings",
        "pre-commit",
    ],
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
    pass


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

    def run_benchmark(self):
        for benchmark_item in glob.glob('tests/benchmark/*py'):
            os.system(f'pytest {benchmark_item}')


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
    version_path = ROOT_DIR / "paddleaudio" / "__init__.py"
    with open(version_path, "a") as f:
        f.write(f"__version__ = '{version}'\n")


def _rm_version():
    file_ = ROOT_DIR / "paddleaudio" / "__init__.py"
    with open(file_, "r") as f:
        lines = f.readlines()
    with open(file_, "w") as f:
        for line in lines:
            if "__version__" not in line:
                f.write(line)


################################# Steup ##################################
class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove paddleaudio extension
        for path in (ROOT_DIR / "paddleaudio").glob("**/*.so"):
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
    _rm_version()

    _make_version_file(version, sha)
    lib_package_data = {}
    if platform.system() != 'Windows' and platform.system() != 'Linux':
        lib_package_data = {'paddleaudio': ['lib/libgcc_s.1.1.dylib']}

    #if platform.system() == 'Linux':
    #    lib_package_data = {'paddleaudio': ['lib/lib*']}

    setup_info = dict(
        # Metadata
        name='paddleaudio',
        version=VERSION,
        author='PaddlePaddle Speech and Language Team',
        author_email='paddlesl@baidu.com',
        url='https://github.com/PaddlePaddle/PaddleSpeech/audio',
        license='Apache 2.0',
        description='Speech audio tools based on Paddlepaddle',
        keywords=[
            "audio process"
            "paddlepaddle",
        ],
        python_requires='>=3.7',
        install_requires=requirements["install"],
        extras_require={
            'develop': requirements["develop"],
            #'test': ["nose", "torchaudio==0.10.2", "pytest-benchmark", "librosa=0.8.1", "parameterized", "paddlepaddle"],
        },
        cmdclass={
            "build_ext": setup_helpers.CMakeBuild,
            'develop': DevelopCommand,
            'test': TestCommand,
            'upload': UploadCommand,
            "clean": clean,
        },

        # Package info
        packages=find_packages(include=('paddleaudio*')),
        package_data=lib_package_data,
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
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ], )

    setup(**setup_info)
    _rm_version()


if __name__ == '__main__':
    main()
