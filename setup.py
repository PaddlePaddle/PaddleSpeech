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
from setuptools import Command
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess as sp

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


VERSION = '2.1.2'
long_description = read("README.md")
deps = [d.strip() for d in read('requirements.txt').split()]


def _pre_install():
    sp.check_call("apt-get update -y".split())
    sp.check_call("apt-get install -y".split() +
                  'vim tig tree sox pkg-config'.split() +
                  'libsndfile1 libflac-dev libogg-dev'.split() +
                  'libvorbis-dev libboost-dev swig python3-dev'.split())


def _post_install(install_lib_dir):
    pass


class DevelopCommand(develop):
    def run(self):
        _pre_install()
        develop.run(self)
        self.execute(_post_install, (self.install_lib, ), msg="Post Install...")


class InstallCommand(install):
    def run(self):
        _pre_install()
        install.run(self)
        self.execute(_post_install, (self.install_lib, ), msg="Post Install...")


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
            shutil.rmtree(os.path.join(HERE, "dist"))
        except OSError:
            pass
        print("Building source distribution...")
        sp.check_call([sys.executable, "setup.py", "sdist"])
        print("Uploading package to PyPi...")
        sp.check_call(["twine", "upload", "dist/*"])
        sys.exit()


setup_info = dict(
    # Metadata
    name='paddle-speech',
    version=VERSION,
    author='PaddleSL Speech Team',
    author_email='',
    url='https://github.com/PaddlePaddle/DeepSpeech',
    license='Apache 2',
    description='Speech tools and models based on Paddlepaddle',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "speech",
        "asr",
        "tts",
        "text frontend",
        "MFA",
        "paddlepaddle",
        "transformer",
        "conformer",
        "fastspeech",
        "vocoder",
        "pwgan",
        "gan",
    ],
    python_requires='>=3.6',
    install_requires=deps,
    extras_require={
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
    packages=find_packages(exclude=('tests', 'tests.*')),
    zip_safe=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)

setup(**setup_info)
