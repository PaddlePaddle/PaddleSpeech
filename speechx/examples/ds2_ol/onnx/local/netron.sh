#!/bin/bash

if [ $# != 1 ];then
    echo "usage: $0 model_path"
    exit 1
fi


file=$1

pip install netron
netron -p 8082 --host $(hostname -i) $file