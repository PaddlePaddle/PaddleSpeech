#!/bin/bash

abort(){
    echo "Run unittest failed" 1>&2
    echo "Please check your code" 1>&2
    exit 1
}

unittest(){
    cd $1 > /dev/null
    if [ -f "setup.sh" ]; then
        sh setup.sh
        export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    fi
    if [ $? != 0 ]; then
        exit 1
    fi
    find . -name 'tests' -type d -print0 | \
        xargs -0 -I{} -n1 bash -c \
        'python -m unittest discover -v -s {}'
    cd - > /dev/null
}

trap 'abort' 0
set -e

unittest .

trap : 0
