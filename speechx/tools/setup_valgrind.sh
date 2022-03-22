#!/bin/bash

VALGRIND_VERSION=3.18.1

set -e

tarball=valgrind-3.18.1.tar.bz2

url=https://sourceware.org/pub/valgrind/valgrind-3.18.1.tar.bz2

if [ -f $tarball ]; then
  echo "use the $tarball have downloaded."
else
  wget -c -t3 --no-check-certificate $url
fi

tar xjfv $tarball

mv valgrind-3.18.1 valgrind

prefix=$PWD/valgrind/install
cd ./valgrind/
  ./configure --prefix=$prefix
  make
  make install
cd -
