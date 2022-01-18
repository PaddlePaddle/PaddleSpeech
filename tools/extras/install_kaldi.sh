#!/bin/bash

# Installation script for Kaldi
#
set -e

apt-get install subversion -y

KALDI_GIT="--depth 1 -b master https://github.com/kaldi-asr/kaldi.git"

KALDI_DIR="$PWD/kaldi"
SHARED=false

if [ ! -d "$KALDI_DIR" ]; then
    git clone $KALDI_GIT $KALDI_DIR
else
    echo "$KALDI_DIR already exists!"
fi

pushd "$KALDI_DIR/tools"
git pull

# Prevent kaldi from switching default python version
mkdir -p "python"
touch "python/.use_default_python"

# check deps
./extras/check_dependencies.sh

# make tools
make -j4

# make src
pushd ../src
OPENBLAS_DIR=${KALDI_DIR}/../OpenBLAS
mkdir -p ${OPENBLAS_DIR}/install
if [ $SHARED == true ]; then
   ./configure --shared --use-cuda=no --static-math --mathlib=OPENBLAS --openblas-root=${OPENBLAS_DIR}/install
else
   ./configure --static --use-cuda=no --static-math --mathlib=OPENBLAS --openblas-root=${OPENBLAS_DIR}/install
fi
make clean -j && make depend -j && make -j4
popd # kaldi/src


popd # kaldi/tools

echo "Done installing Kaldi."
