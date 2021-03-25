#!/bin/bash

unittest(){
    cd $1 > /dev/null
    find . -path ./tools/venv -prune -false -o -name 'tests' -type d -print0 | \
        xargs -0 -I{} -n1 bash -c \
        'python3 -m unittest discover -v -s {}'
    cd - > /dev/null
}

abort(){
    echo "Run unittest failed" 1>&2
    echo "Please check your code" 1>&2
    exit 1
}

trap 'abort' 0
set -e

source tools/venv/bin/activate
pip3 install pytest
unittest .

trap : 0
