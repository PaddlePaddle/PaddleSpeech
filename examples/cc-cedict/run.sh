#!/bin/bash

# CC-CEDICT download: https://www.mdbg.net/chinese/dictionary?page=cc-cedict
# The word dictionary of this website is based on CC-CEDICT.
# CC-CEDICT is a continuation of the CEDICT project started by Paul Denisowski in 1997 with the
# aim to provide a complete downloadable Chinese to English dictionary with pronunciation in pinyin for the Chinese characters.
# This website allows you to easily add new entries or correct existing entries in CC-CEDICT.
# Submitted entries will be checked and processed frequently and released for download in CEDICT format on this page.

set -e
source path.sh

stage=-1
stop_stage=100


source ${MAIN_ROOT}/utils/parse_options.sh || exit -1


cedict_url=https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.zip
cedict=cedict_1_0_ts_utf-8_mdbg.zip

mkdir -p data

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ];then
    test -f data/${cedict} || wget -O data/${cedict} ${cedict_url}
    pushd data
    unzip ${cedict}
    popd

fi

mkdir -p exp

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
    cp data/cedict_ts.u8 exp/cedict
    python3 local/parser.py exp/cedict exp/cedict.json
fi

