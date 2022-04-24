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
import glob
import os
import subprocess

import pybind11
import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.test import test

# set the version here
VERSION = '1.0.0a'


# Inspired by the example at https://pytest.org/latest/goodpractises.html
class TestCommand(test):
    def finalize_options(self):
        test.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run(self):
        self.run_benchmark()
        super(TestCommand, self).run()

    def run_tests(self):
        # Run nose ensuring that argv simulates running nosetests directly
        import nose
        nose.run_exit(argv=['nosetests', '-w', 'tests'])

    def run_benchmark(self):
        for benchmark_item in glob.glob('tests/benchmark/*py'):
            os.system(f'pytest {benchmark_item}')


class InstallCommand(install):
    def run(self):
        install.run(self)


class ExtBuildCommand(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        super().run()

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Debug" if self.debug else "Release"
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            "-DBUILD_SOX:BOOL=ON",
        ]
        build_args = ["--target", "install"]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", os.path.abspath(os.path.dirname(__file__))] + cmake_args,
            cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp)

    def get_ext_filename(self, fullname):
        ext_filename = super().get_ext_filename(fullname)
        ext_filename_parts = ext_filename.split(".")
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        ext_filename = ".".join(without_abi)
        return ext_filename


def write_version_py(filename='paddleaudio/__init__.py'):
    with open(filename, "a") as f:
        f.write(f"__version__ = '{VERSION}'")


def remove_version_py(filename='paddleaudio/__init__.py'):
    with open(filename, "r") as f:
        lines = f.readlines()
    with open(filename, "w") as f:
        for line in lines:
            if "__version__" not in line:
                f.write(line)


def get_ext_modules():
    modules = [
        Extension(name="paddleaudio._paddleaudio", sources=[]),
    ]

    return modules


remove_version_py()
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
        'numpy >= 1.15.0', 'scipy >= 1.0.0', 'resampy >= 0.2.2',
        'soundfile >= 0.9.0', 'colorlog', 'dtaidistance == 2.3.1', 'pathos'
    ],
    extras_require={
        'test': [
            'nose', 'librosa==0.8.1', 'soundfile==0.10.3.post1',
            'torchaudio==0.10.2', 'pytest-benchmark'
        ],
    },
    ext_modules=get_ext_modules(),
    cmdclass={
        'install': InstallCommand,
        "build_ext": ExtBuildCommand,
        'test': TestCommand,
    }, )

remove_version_py()
