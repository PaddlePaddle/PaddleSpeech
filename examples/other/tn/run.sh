#!/bin/bash

source path.sh

USE_SCLITE=true

# test text normalization
echo "Start get text normalization test data ..."
python3 get_textnorm_data.py --test-file=data/textnorm_test_cases.txt --output-dir=data/textnorm
echo "Start test text normalization ..."
python3 test_textnorm.py --input-dir=data/textnorm --output-dir=exp/textnorm

# whether use sclite to get more detail information of WER
if [ "$USE_SCLITE" = true ];then
    echo "Start sclite textnorm ..."
    ${MAIN_ROOT}/tools/sctk/bin/sclite -i wsj -r ./exp/textnorm/text.ref.clean trn -h ./exp/textnorm/text.tn trn -e utf-8 -o all
fi