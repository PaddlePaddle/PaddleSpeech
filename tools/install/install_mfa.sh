#!/bin/bash

test -d Montreal-Forced-Aligner || git clone https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git

pushd Montreal-Forced-Aligner && python setup.py install

test -d kaldi || { echo "need install kaldi first"; exit 1;}

mfa thirdparty kaldi $PWD/kaldi

mfa thirdparty validate

echo "install mfa pass."
