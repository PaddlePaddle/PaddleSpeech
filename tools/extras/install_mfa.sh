#!/bin/bash

# install openblas, kaldi before

test -d Montreal-Forced-Aligner || git clone https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git

pushd Montreal-Forced-Aligner && git checkout v2.0.0a7 &&  python setup.py install

test -d kaldi || { echo "need install kaldi first"; exit 1;}

mfa thirdparty kaldi $PWD/kaldi

mfa thirdparty validate

echo "install mfa pass."
