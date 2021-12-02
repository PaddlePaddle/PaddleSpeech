#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#           2021 PaddlePaddle
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <set> <lang>>"
    echo "e.g.: $0 dev"
    exit 1
fi

set=$1
lang=$2
export LC_ALL=en_US.UTF-8
# Copy stuff intoc its final locations [this has been moved from the format_data script]
# for En
mkdir -p ${set}.en
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f ${set}/${f} ]; then
        sort ${set}/${f} > ${set}.en/${f}
    fi
done
sort ${set}/text.en | sed $'s/[^[:print:]]//g' > ${set}.en/text 

utils/fix_data_dir.sh ${set}.en
if [ -f ${set}.en/feats.scp ]; then
    utils/validate_data_dir.sh ${set}.en || exit 1;
else
    utils/validate_data_dir.sh --no-feats --no-wav ${set}.en || exit 1;
fi

# for target language
mkdir -p ${set}.${lang}
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f ${set}/${f} ]; then
        sort ${set}/${f} > ${set}.${lang}/${f}
    fi
done
sort ${set}/text.${lang} | sed $'s/[^[:print:]]//g' > ${set}.${lang}/text 
utils/fix_data_dir.sh  ${set}.${lang}
if [ -f ${set}.${lang}/feats.scp ]; then
    utils/validate_data_dir.sh ${set}.${lang} || exit 1;
else
    utils/validate_data_dir.sh --no-feats --no-wav ${set}.${lang} || exit 1;
fi
