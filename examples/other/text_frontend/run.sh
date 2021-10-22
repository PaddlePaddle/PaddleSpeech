#!/bin/bash

USE_SCLITE=true

# test g2p
echo "Start get g2p test data ..."
python3 get_g2p_data.py --root-dir=~/datasets/BZNSYP --output-dir=data/g2p
echo "Start test g2p ..."
python3 test_g2p.py --input-dir=data/g2p --output-dir=exp/g2p

# test text normalization
echo "Start get text normalization test data ..."
python3 get_textnorm_data.py --test-file=data/textnorm_test_cases.txt --output-dir=data/textnorm
echo "Start test text normalization ..."
python3 test_textnorm.py --input-dir=data/textnorm --output-dir=exp/textnorm

# whether use sclite to get more detail information of WER
if [ "$USE_SCLITE" = true ];then
    echo "Start sclite g2p ..."
    ./SCTK/bin/sclite -i wsj -r ./exp/g2p/text.ref.clean trn -h ./exp/g2p/text.g2p trn -e utf-8 -o all
    echo

    echo "Start sclite textnorm ..."
    ./SCTK/bin/sclite -i wsj -r ./exp/textnorm/text.ref.clean trn -h ./exp/textnorm/text.tn trn -e utf-8 -o all
fi