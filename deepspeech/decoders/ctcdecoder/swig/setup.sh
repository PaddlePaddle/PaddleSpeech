#!/usr/bin/env bash

if [ ! -d kenlm ]; then
    git clone https://github.com/kpu/kenlm.git
    cd kenlm/
    git checkout df2d717e95183f79a90b2fa6e4307083a351ca6a
    cd ..
    echo -e "\n"
fi

if [ ! -d openfst-1.6.3 ]; then
    echo "Download and extract openfst ..."
    wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz --no-check-certificate
    tar -xzvf openfst-1.6.3.tar.gz
    echo -e "\n"
fi

if [ ! -d ThreadPool ]; then
    git clone https://github.com/progschj/ThreadPool.git
    echo -e "\n"
fi

echo "Install decoders ..."
python3 setup.py install --num_processes 4
