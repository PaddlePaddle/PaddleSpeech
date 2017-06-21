#!/bin/bash

# install python dependencies
if [ -f 'requirements.txt' ]; then
    pip install -r requirements.txt
fi
if [ $? != 0 ]; then
    echo "Install python dependencies failed !!!"
    exit 1
fi

# install scikits.samplerate
curl -O "http://www.mega-nerd.com/SRC/libsamplerate-0.1.9.tar.gz"
if [ $? != 0 ]; then
    echo "Download libsamplerate-0.1.9.tar.gz failed !!!"
    exit 1
fi
tar -xvf libsamplerate-0.1.9.tar.gz
cd libsamplerate-0.1.9
./configure && make && make install
cd -
rm -rf libsamplerate-0.1.9
rm libsamplerate-0.1.9.tar.gz
pip install scikits.samplerate==0.3.3
if [ $? != 0 ]; then
    echo "Install scikits.samplerate failed !!!"
    exit 1
fi

# prepare ./checkpoints
mkdir checkpoints

echo "Install all dependencies successfully."
