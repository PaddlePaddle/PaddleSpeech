#!/usr/bin/env bash

set -e

source path.sh


stage=0
stop_stage=100
# bpemode (unigram or bpe)
nbpe=100
bpemode=unigram


source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

train_set=train
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/

    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1

    # we borrowed these code and scripts which are related bpe from ESPnet.
    cut -f 2- -d" " text > data/lang_char/input.txt
    ${MAIN_ROOT}/utils/spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    ${MAIN_ROOT}/utils/spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
    wc -l ${dict}
fi

${MAIN_ROOT}/utils/spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt > data/lang_char/input.bpe
${MAIN_ROOT}/utils/spm_decode --model=${bpemodel}.model --input_format=piece < data/lang_char/input.bpe | sed -e "s/▁/ /g" > data/lang_char/input.decode
