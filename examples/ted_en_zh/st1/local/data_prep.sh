#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

data_dir=${1}

for set in train dev test; do
# for set in train; do
    dst=${target_dir}/${set}
    for lang in en zh; do

        if [ ${lang} = 'en' ]; then
            echo "remove punctuation $lang"
            # remove punctuation
            local/remove_punctuation.pl < ${dst}/${lang}.org > ${dst}/${lang}.raw
        else
            cp ${dst}/${lang}.org ${dst}/${lang}.raw
        fi

        paste -d " " ${dst}/.yaml ${dst}/${lang}.raw | sort > ${dst}/text.${lang}


    done
    # error check
    n=$(cat ${dst}/.yaml | wc -l)
    n_en=$(cat ${dst}/en.raw | wc -l)
    n_tgt=$(cat ${dst}/zh.raw | wc -l)
    [ ${n} -ne ${n_en} ] && echo "Warning: expected ${n} data data files, found ${n_en}" && exit 1;
    [ ${n} -ne ${n_tgt} ] && echo "Warning: expected ${n} data data files, found ${n_tgt}" && exit 1;

    echo "done text processing"
    cat ${dst}/wav.scp.org | uniq | sort -k1,1 -u > ${dst}/wav.scp
    cat ${dst}/utt2spk.org | uniq | sort -k1,1 -u > ${dst}/utt2spk

    cat ${dst}/utt2spk | utt2spk_to_spk2utt.pl | sort -k1,1 -u > ${dst}/spk2utt
    rm -rf ${target_dir}/data_prep/${set}.en-zh
    mkdir -p ${target_dir}/data_prep/${set}.en-zh
    echo "remove duplicate lines..."
    cut -d ' ' -f 1 ${dst}/text.en | sort | uniq -c | sort -n -k1 -r | grep -v '1 ted-en-zh' \
        | sed 's/^[ \t]*//' > ${dst}/duplicate_lines
    cut -d ' ' -f 1 ${dst}/text.en | sort | uniq -c | sort -n -k1 -r | grep '1 ted-en-zh' \
        | cut -d '1' -f 2- | sed 's/^[ \t]*//' > ${dst}/reclist
    reduce_data_dir.sh ${dst} ${dst}/reclist ${target_dir}/data_prep/${set}.en-zh
    echo "done wav processing"
    for l in en zh; do
        cp ${dst}/text.${l} ${target_dir}/data_prep/${set}.en-zh/text.${l}
    done
    fix_data_dir.sh --utt_extra_files \
    "text.en text.zh" \
    ${target_dir}/data_prep/${set}.en-zh
done      