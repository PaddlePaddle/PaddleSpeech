#!/bin/bash

if [ ! -d "./SCTK" ];then
    echo "Clone SCTK ..."
    git clone https://github.com/usnistgov/SCTK
    echo "Clone SCTK done!"
fi

if [ ! -d "./SCTK/bin" ];then
    echo "Start make SCTK ..."
    pushd SCTK && make config && make all && make check && make install && make doc && popd
    echo "SCTK make done!"
fi
