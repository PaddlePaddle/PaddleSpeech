#!/bin/bash

# install kaldi-comptiable feature
pushd python_kaldi_features
python3 setup.py install
if [ $? != 0 ]; then
   error_msg "Please check why kaldi feature install error!"
   exit -1
fi
popd
