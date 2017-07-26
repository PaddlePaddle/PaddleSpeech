#!/bin/bash

# install python dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
if [ $? != 0 ]; then
    echo "Install python dependencies failed !!!"
    exit 1
fi

# install package libsndfile
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
mkdir thirdparty
curl -O "http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz"
if [ $? != 0 ]; then
    echo "Download libsndfile-1.0.28.tar.gz failed !!!"
    exit 1
fi
tar -zxvf libsndfile-1.0.28.tar.gz
cd libsndfile-1.0.28
./configure --prefix=$DIR/thirdparty/libsndfile && make && make install
cd ..
rm -rf libsndfile-1.0.28
rm libsndfile-1.0.28.tar.gz

# prepare ./checkpoints
mkdir checkpoints

echo "Install all dependencies successfully."
