#!/bin/bash



abort(){
    echo "Run unittest failed" 1>&2
    echo "Please check your code" 1>&2
    exit 1
}


unittest(){
    cd $1 > /dev/null
    if [ -f "setup.sh" ]; then
        bash setup.sh
        export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    fi
    if [ $? != 0 ]; then
        exit 1
    fi
    find . -path ./tools/venv -prune -false -o -name 'tests' -type d -print0 | \
        xargs -0 -I{} -n1 bash -c \
        'python3 -m unittest discover -v -s {}'
    cd - > /dev/null
}

coverage(){
    cd $1 > /dev/null

    if [ -f "setup.sh" ]; then
        bash setup.sh
        export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    fi
    if [ $? != 0 ]; then
        exit 1
    fi

    find . -path ./tools/venv -prune -false -o -name 'tests' -type d -print0 | \
        xargs -0 -I{} -n1 bash -c \
        'python3 -m coverage run --branch {}'
    python3 -m coverage report -m
    python3 -m coverage html
    cd - > /dev/null
}

trap 'abort' 0
set -e

source tools/venv/bin/activate
#pip3 install pytest
#unittest .
coverage .

trap : 0
