#!/bin/bash
set -eo pipefail

. path.sh

stage=-1
stop_stage=100
corpus=aishell

unit=data/vocab.txt     # vocab
lexicon=  # aishell/resource_aishell/lexicon.txt
text=     # aishell/data_aishell/transcript/aishell_transcript_v0.8.txt

. parse_options.sh

data=$PWD/data
mkdir -p $data

if [ ! -f $unit ]; then
    echo "$0: No such file $unit"
    exit 1;
fi

if [ ! which ngram-count ]; then
    pushd $MAIN_ROOT/tools
    make srilm.done
    popd
fi

if [ ! which fstaddselfloops ]; then
    pushd $MAIN_ROOT/tools
    make kaldi.done
    popd
fi

mkdir -p data/local/dict
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # 7.1 Prepare dict
    cp $unit data/local/dict/units.txt
    utils/fst/prepare_dict.py \
        --unit_file $unit \
        --in_lexicon ${lexicon} \
        --out_lexicon data/local/dict/lexicon.txt
fi

lm=data/local/lm
mkdir -p data/train
mkdir -p $lm
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # 7.2 Train lm
    utils/manifest_key_value.py \
        --manifest_path data/manifest.train \
        --output_path data/train
    utils/filter_scp.pl data/train/text \
        $text > $lm/text
    
    local/aishell_train_lms.sh
fi

echo "build LM done."
exit 0
