#! /usr/bin/env  bash

SUDO='sudo'
if [ $(id -u) -eq 0 ]; then
  SUDO=''
fi

if [ -e /etc/lsb-release ];then
    #${SUDO} apt-get update
    ${SUDO} apt-get install -y pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig python3-dev
fi

# install python dependencies
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
fi
if [ $? != 0 ]; then
    echo "Install python dependencies failed !!!"
    exit 1
fi

# install package libsndfile
python3 -c "import soundfile"
if [ $? != 0 ]; then
    echo "Install package libsndfile into default system path."
    wget "http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz"
    if [ $? != 0 ]; then
        echo "Download libsndfile-1.0.28.tar.gz failed !!!"
        exit 1
    fi
    tar -zxvf libsndfile-1.0.28.tar.gz
    cd libsndfile-1.0.28
    ./configure > /dev/null && make > /dev/null && make install > /dev/null
    cd ..
    rm -rf libsndfile-1.0.28
    rm libsndfile-1.0.28.tar.gz
fi

# install decoders
python3 -c "import pkg_resources; pkg_resources.require(\"swig_decoders==1.1\")"
if [ $? != 0 ]; then
    cd deepspeech/decoders/swig > /dev/null
    sh setup.sh
    cd - > /dev/null
fi


echo "Install all dependencies successfully."
