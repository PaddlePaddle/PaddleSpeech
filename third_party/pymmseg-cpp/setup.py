#!/usr/bin/env python3
from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

VERSION_INFO = (1, 2, 0)
DATE_INFO = (2013, 2, 10)  # YEAR, MONTH, DAY
VERSION = '.'.join(str(i) for i in VERSION_INFO)
REVISION = '%04d%02d%02d' % DATE_INFO
BUILD_INFO = "MMSeg v" + VERSION + " (" + REVISION + ")"
AUTHOR = "pluskid & kronuz & zsp007"
AUTHOR_EMAIL = 'pluskid@gmail.com'
URL = 'http://github.com/pluskid/pymmseg-cpp'
DOWNLOAD_URL = 'https://github.com/pluskid/pymmseg-cpp/archive/master.tar.gz'
LICENSE = "MIT"
PROJECT = "pymmseg"


def read(fname):
    import os
    try:
        return open(os.path.join(os.path.dirname(__file__),
                                 fname)).read().strip()
    except IOError:
        return ''


extra = {}
import sys
if sys.version_info >= (3, 0):
    extra.update(use_2to3=True, )

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(
        "mmseg",
        [
            'mmseg/mmseg-cpp/mmseg.cpp', 'mmseg/mmseg-cpp/algor.cpp',
            'mmseg/mmseg-cpp/dict.cpp', 'mmseg/mmseg-cpp/memory.cpp'
        ],
        include_dirs=['mmseg/mmseg-cpp'],
        # Example: passing in the version to the compiled code
        define_macros=[('VERSION_INFO', VERSION_INFO)],
    ),
]

setup(
    name=PROJECT,
    version=VERSION,
    description=read('DESCRIPTION'),
    long_description=read('README'),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    keywords='mmseg chinese word segmentation tokenization',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", "Programming Language :: Python",
        "Programming Language :: Python :: 3", "Topic :: Text Processing",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    setup_requires=["pybind11"],
    install_requires=["pybind11"],
    #packages=['mmseg'],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    package_data={'mmseg': ['data/*.dic']},
    scripts=['bin/pymmseg'],
    **extra)
