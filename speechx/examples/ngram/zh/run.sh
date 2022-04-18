#!/bin/bash
set -eo pipefail

. path.sh

stage=-1
stop_stage=100
corpus=aishell

unit=data/vocab.txt       # vocab file, line: char/spm_pice
lexicon=data/lexicon.txt  # line: word ph0 ... phn, aishell/resource_aishell/lexicon.txt
text=data/text            # line: utt text, aishell/data_aishell/transcript/aishell_transcript_v0.8.txt

. utils/parse_options.sh

data=$PWD/data
mkdir -p $data

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
    if [ ! -f $data/speech.ngram.zh.tar.gz ];then
        pushd $data
        wget -c http://paddlespeech.bj.bcebos.com/speechx/examples/ngram/zh/speech.ngram.zh.tar.gz
        tar xvzf speech.ngram.zh.tar.gz
        popd
    fi
fi

if [ ! -f $unit ]; then
    echo "$0: No such file $unit"
    exit 1;
fi

if ! which ngram-count; then
    pushd $MAIN_ROOT/tools
    make srilm.done
    popd
fi

mkdir -p data/local/dict
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # 7.1 Prepare dict
    # line: char/spm_pices
    cp $unit data/local/dict/units.txt

    if [ ! -f $lexicon ];then
        local/text_to_lexicon.py --has_key true --text $text --lexicon $lexicon
        echo "Generate $lexicon from $text"
    fi

    # filter by vocab
    # line: word ph0 ... phn -> line: word char0 ... charn
    utils/fst/prepare_dict.py \
        --unit_file $unit \
        --in_lexicon ${lexicon} \
        --out_lexicon data/local/dict/lexicon.txt
fi

lm=data/local/lm
mkdir -p $lm

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # 7.2 Train lm
    cp $text $lm/text
    local/aishell_train_lms.sh
fi

echo "build LM done."
exit 0
