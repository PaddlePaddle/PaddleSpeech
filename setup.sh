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
python -c "import soundfile"
if [ $? != 0 ]; then
    echo "Install package libsndfile into default system path."
    curl -O "http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz"
    if [ $? != 0 ]; then
        echo "Download libsndfile-1.0.28.tar.gz failed !!!"
        exit 1
    fi
    tar -zxvf libsndfile-1.0.28.tar.gz
    cd libsndfile-1.0.28
    ./configure && make && make install
    cd ..
    rm -rf libsndfile-1.0.28
    rm libsndfile-1.0.28.tar.gz
fi

echo "Install all dependencies successfully."
