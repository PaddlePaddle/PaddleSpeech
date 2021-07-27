#!/bin/bash

apt install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev

apt-get install -y gcc-5 g++-5 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50  && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50

test -d kenlm || wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz

rm -rf kenlm/build && mkdir -p kenlm/build && cd kenlm/build && cmake .. && make -j4 && make install
