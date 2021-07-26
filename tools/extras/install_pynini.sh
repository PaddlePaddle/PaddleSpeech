#!/bin/bash

set -e
set -x

pynini=pynini-2.1.4
openfst=openfst-1.8.1

LIBRARY_PATH=$PWD/${openfst}/install/lib

test -e ${pynini}.tar.gz || wget http://www.openfst.org/twiki/pub/GRM/PyniniDownload/${pynini}.tar.gz
test -d ${pynini} || tar -xvf ${pynini}.tar.gz && chown -R root:root ${pynini}

pushd ${pynini} &&  LIBRARY_PATH=$LIBRARY_PATH  python setup.py install && popd
