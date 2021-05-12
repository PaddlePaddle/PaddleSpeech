#!/bin/bash

setup_env(){
    cd tools && make && cd - 
}

install(){
    if [ -f "setup.sh" ]; then
        bash setup.sh
        #export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    fi
    if [ $? != 0 ]; then
        exit 1
    fi
}

print_env(){
    cat /etc/lsb-release
    gcc -v
    g++ -v
}

abort(){
    echo "Run install failed" 1>&2
    echo "Please check your code" 1>&2
    exit 1
}

trap 'abort' 0
set -e

print_env
setup_env
source tools/venv/bin/activate
install

trap : 0
