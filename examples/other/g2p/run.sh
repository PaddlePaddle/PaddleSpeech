#!/bin/bash

source path.sh
USE_SCLITE=true

# test g2p
if [ ! -d ~/datasets/BZNSYP ];then
    echo "Please download BZNSYP dataset"
    exit
fi
echo "Start get g2p test data ..."
python3 get_g2p_data.py --root-dir=~/datasets/BZNSYP --output-dir=data/g2p
echo "Start test g2p ..."
python3 test_g2p.py --input-dir=data/g2p --output-dir=exp/g2p

# whether use sclite to get more detail information of WER
if [ "$USE_SCLITE" = true ];then
    echo "Start sclite g2p ..."
    ${MAIN_ROOT}/tools/sctk/bin/sclite -i wsj -r ./exp/g2p/text.ref.clean trn -h ./exp/g2p/text.g2p trn -e utf-8 -o all
fi
