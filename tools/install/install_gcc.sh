#!/bin/bash

set -e
set -x

# gcc
apt update -y
apt install build-essential -y
apt install software-properties-common -y
add-apt-repository ppa:ubuntu-toolchain-r/test
apt install gcc-8 g++-8 -y
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80
update-alternatives --config gcc

# gfortran
apt-get install gfortran-8
