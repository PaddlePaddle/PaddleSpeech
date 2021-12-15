#!/bin/bash

if [ ! -d data ]; then
    wget -c https://paddlespeech.bj.bcebos.com/datasets/iwslt2012.tar.gz
    tar -xzf iwslt2012.tar.gz
fi

echo "Finish data preparation."
exit 0
