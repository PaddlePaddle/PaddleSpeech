#!/bin/bash

FILEID=0B4y35FiV1wh7QVR6VXJ5dWExSTQ
FILENAME=CRF++-0.58

test -e ${FILENAME} || wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O ${FILENAME}.tar.gz

tar zxvf ${FILENAME}.tar.gz

pushd ${FILENAME}
./configure
make -j
make install
popd

pushd ${FILENAME}/python
python3 setup.py build
python3 setup.py install
popd

which crf_learn