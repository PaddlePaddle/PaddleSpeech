#!/bin/bash
set -eo pipefail

. path.sh

stage=-1
stop_stage=100
corpus=aishell
lmtype=srilm

lexicon=  # aishell/resource_aishell/lexicon.txt
text=     # aishell/data_aishell/transcript/aishell_transcript_v0.8.txt

source parse_options.sh

if [ ! which fstprint ]; then
    pushd $MAIN_ROOT/tools
    make kaldi.done
    popd
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # 7.1 Prepare dict
    unit_file=data/vocab.txt
    mkdir -p data/local/dict
    cp $unit_file data/local/dict/units.txt
    utils/fst/prepare_dict.py \
        --unit_file $unit_file \
        --in_lexicon ${lexicon} \
        --out_lexicon data/local/dict/lexicon.txt
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # 7.2 Train lm
    lm=data/local/lm
    mkdir -p data/train
    mkdir -p $lm
    utils/manifest_key_value.py \
        --manifest_path data/manifest.train \
        --output_path data/train
    utils/filter_scp.pl data/train/text \
        $text > $lm/text
    if [ $lmtype == 'srilm' ];then
        local/aishell_train_lms.sh
    else
        utils/ngram_train.sh --order 3 $lm/text $lm/lm.arpa
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then 
    # 7.3 Build decoding TLG
    utils/fst/compile_lexicon_token_fst.sh \
        data/local/dict data/local/tmp data/local/lang
    utils/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;
fi

echo "Aishell build TLG done."
exit 0
