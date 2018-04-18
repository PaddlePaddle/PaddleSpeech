#!/usr/bin/env bash

if [ ! -d kenlm ]; then
    git clone https://github.com/luotao1/kenlm.git
    echo -e "\n"
fi

if [ ! -d openfst-1.6.7 ]; then
    echo "Download and extract openfst ..."
    wget https://sites.google.com/site/openfst/home/openfst-down/openfst-1.6.7.tar.gz
    tar -xzvf openfst-1.6.7.tar.gz
    echo -e "\n"
fi

if [ ! -d ThreadPool ]; then
    git clone https://github.com/progschj/ThreadPool.git
    echo -e "\n"
fi

echo "Install decoders ..."
python setup.py install --num_processes 4
