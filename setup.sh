#! /usr/bin/env  bash

source utils/log.sh


SUDO='sudo'
if [ $(id -u) -eq 0 ]; then
  SUDO=''
fi

if [ -e /etc/lsb-release ]; then
    ${SUDO} apt-get install -y pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig python3-dev
else
    error_msg "Please using Ubuntu or install pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig python3-dev by user."
    exit -1
fi

# install python dependencies
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
fi
if [ $? != 0 ]; then
    error_msg "Install python dependencies failed !!!"
    exit 1
fi

# install package libsndfile
python3 -c "import soundfile"
if [ $? != 0 ]; then
    info_msg "Install package libsndfile into default system path."
    wget "http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.28.tar.gz"
    if [ $? != 0 ]; then
        error_msg "Download libsndfile-1.0.28.tar.gz failed !!!"
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
python3 -c "import pkg_resources; pkg_resources.require(\"swig_decoders==1.1\")"
if [ $? != 0 ]; then
   error_msg "Please check why decoder install error!"
   exit -1
fi

info_msg "Install all dependencies successfully."
