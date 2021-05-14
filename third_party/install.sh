#!/bin/bash

# install kaldi-comptiable feature
pushd python_kaldi_features
python3 setup.py install
if [ $? != 0 ]; then
   error_msg "Please check why kaldi feature install error!"
   exit -1
fi
popd

# install zhon
pushd zhon
python3 setup.py install
if [ $? != 0 ]; then
   error_msg "Please check why zhon install error!"
   exit -1
fi
popd

# install pypinyin
pushd python-pinyin
python3 setup.py install
if [ $? != 0 ]; then
   error_msg "Please check why pypinyin install error!"
   exit -1
fi
popd

# install mmseg
pushd pymmseg-cpp/
python3 setup.py install
if [ $? != 0 ]; then
   error_msg "Please check why pymmseg install error!"
   exit -1
fi
popd

