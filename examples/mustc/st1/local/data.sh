#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#           2021 PaddlePaddle
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
set -u

stage=-1
stop_stage=10

# bpemode (unigram or bpe)
tgt_lang=
nbpe=8000
bpemode=bpe
must_c=
dumpdir=data/dump
do_delta=false
tgt_case=tc
src_case=lc.rm
source ${MAIN_ROOT}/utils/parse_options.sh

TARGET_DIR=${MAIN_ROOT}/examples/dataset
mkdir -p ${TARGET_DIR}
mkdir -p data

train_set=train_sp.en-${tgt_lang}.${tgt_lang}
train_dev=dev.en-${tgt_lang}.${tgt_lang}
trans_set=""
for lang in $(echo ${tgt_lang} | tr '_' ' '); do
    trans_set="${trans_set} tst-COMMON.en-${lang}.${lang}"
done


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ ! -e ${must_c} ]; then
        echo "Error: Dataset is not avaiable. Please download and unzip the dataset"
        echo "Link of Must-c v1, https://ict.fbk.eu/must-c/."
        exit 1
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data Preparation"
    for lang in $(echo ${tgt_lang} | tr '_' ' '); do
        local/data_prep.sh ${must_c} ${lang}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for lang in $(echo ${tgt_lang} | tr '_' ' '); do
        for x in train.en-${tgt_lang} dev.en-${tgt_lang} tst-COMMON.en-${tgt_lang} tst-HE.en-${tgt_lang}; do
            steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
                data/${x} data/make_fbank/${x} ${fbankdir}
        done
    done

    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 data/train.en-${tgt_lang} data/temp1.${tgt_lang}
    utils/perturb_data_dir_speed.sh 1.0 data/train.en-${tgt_lang} data/temp2.${tgt_lang}
    utils/perturb_data_dir_speed.sh 1.1 data/train.en-${tgt_lang} data/temp3.${tgt_lang}
    utils/combine_data.sh --extra-files utt2uniq data/train_sp.en-${tgt_lang} \
        data/temp1.${tgt_lang} data/temp2.${tgt_lang} data/temp3.${tgt_lang}
    rm -r data/temp1.${tgt_lang} data/temp2.${tgt_lang} data/temp3.${tgt_lang}
    utils/fix_data_dir.sh data/train_sp.en-${tgt_lang}
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        data/train_sp.en-${tgt_lang} data/make_fbank/train_sp.en-${tgt_lang} ${fbankdir}
    for lang in en ${tgt_lang}; do
        awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' data/train.en-${tgt_lang}/utt2spk > data/train_sp.en-${tgt_lang}/utt_map
        utils/apply_map.pl -f 1 data/train_sp.en-${tgt_lang}/utt_map <data/train.en-${tgt_lang}/text.tc.${lang} >data/train_sp.en-${tgt_lang}/text.tc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${tgt_lang}/utt_map <data/train.en-${tgt_lang}/text.lc.${lang} >data/train_sp.en-${tgt_lang}/text.lc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${tgt_lang}/utt_map <data/train.en-${tgt_lang}/text.lc.rm.${lang} >data/train_sp.en-${tgt_lang}/text.lc.rm.${lang}
        awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' data/train.en-${tgt_lang}/utt2spk > data/train_sp.en-${tgt_lang}/utt_map
        utils/apply_map.pl -f 1 data/train_sp.en-${tgt_lang}/utt_map <data/train.en-${tgt_lang}/text.tc.${lang} >>data/train_sp.en-${tgt_lang}/text.tc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${tgt_lang}/utt_map <data/train.en-${tgt_lang}/text.lc.${lang} >>data/train_sp.en-${tgt_lang}/text.lc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${tgt_lang}/utt_map <data/train.en-${tgt_lang}/text.lc.rm.${lang} >>data/train_sp.en-${tgt_lang}/text.lc.rm.${lang}
        awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' data/train.en-${tgt_lang}/utt2spk > data/train_sp.en-${tgt_lang}/utt_map
        utils/apply_map.pl -f 1 data/train_sp.en-${tgt_lang}/utt_map <data/train.en-${tgt_lang}/text.tc.${lang} >>data/train_sp.en-${tgt_lang}/text.tc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${tgt_lang}/utt_map <data/train.en-${tgt_lang}/text.lc.${lang} >>data/train_sp.en-${tgt_lang}/text.lc.${lang}
        utils/apply_map.pl -f 1 data/train_sp.en-${tgt_lang}/utt_map <data/train.en-${tgt_lang}/text.lc.rm.${lang} >>data/train_sp.en-${tgt_lang}/text.lc.rm.${lang}
    done

    # Divide into source and target languages
    for x in train_sp.en-${tgt_lang} dev.en-${tgt_lang} tst-COMMON.en-${tgt_lang} tst-HE.en-${tgt_lang}; do
        local/divide_lang.sh ${x} ${tgt_lang}
    done

    for x in train_sp.en-${tgt_lang} dev.en-${tgt_lang}; do
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        for lang in ${tgt_lang} en; do
            remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}.${lang} data/${x}.${lang}.tmp
        done

        # Match the number of utterances between source and target languages
        # extract commocn lines
        cut -f 1 -d " " data/${x}.en.tmp/text > data/${x}.${tgt_lang}.tmp/reclist1
        cut -f 1 -d " " data/${x}.${tgt_lang}.tmp/text > data/${x}.${tgt_lang}.tmp/reclist2
        comm -12 data/${x}.${tgt_lang}.tmp/reclist1 data/${x}.${tgt_lang}.tmp/reclist2 > data/${x}.en.tmp/reclist

        for lang in ${tgt_lang} en; do
            reduce_data_dir.sh data/${x}.${lang}.tmp data/${x}.en.tmp/reclist data/${x}.${lang}
            utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${x}.${lang}
        done
        rm -rf data/${x}.*.tmp
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/must_c/st1/dump/${train_set}/delta${do_delta}/storage \
          ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{14,15,16,17}/${USER}/espnet-data/egs/must_c/st1/dump/${train_dev}/delta${do_delta}/storage \
          ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark data/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark data/dump_feats/${train_dev} ${feat_dt_dir}
    for ttask in ${trans_set}; do
        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}; mkdir -p ${feat_trans_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${ttask}/feats.scp data/${train_set}/cmvn.ark data/dump_feats/trans/${ttask} \
            ${feat_trans_dir}
    done
fi

dict=data/lang_1spm/${train_set}_${bpemode}${nbpe}_units_${tgt_case}.txt
nlsyms=data/lang_1spm/${train_set}_non_lang_syms_${tgt_case}.txt
bpemodel=data/lang_1spm/${train_set}_${bpemode}${nbpe}_${tgt_case}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1spm/
    export LC_ALL=C.UTF-8

    echo "make a non-linguistic symbol list for all languages"
    grep sp1.0 data/train_sp.en-${tgt_lang}.*/text.${tgt_case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a joint source and target dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    offset=$(wc -l < ${dict})
    grep sp1.0 data/train_sp.en-${tgt_lang}.${tgt_lang}/text.${tgt_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' > data/lang_1spm/input_${tgt_lang}.txt
    grep sp1.0 data/train_sp.en-${tgt_lang}.en/text.${src_case} | cut -f 2- -d' ' | grep -v -e '^\s*$' >> data/lang_1spm/input_${tgt_lang}.txt
    spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_1spm/input_${tgt_lang}.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_1spm/input_${tgt_lang}.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text data/${train_set}/text.${tgt_case} --bpecode ${bpemodel}.model --lang ${tgt_lang} \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.${tgt_case}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --text data/${train_dev}/text.${tgt_case} --bpecode ${bpemodel}.model --lang ${tgt_lang} \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.${tgt_case}.json
    for ttask in ${trans_set}; do
        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}
        data2json.sh --feat ${feat_trans_dir}/feats.scp --text data/${ttask}/text.${tgt_case} --bpecode ${bpemodel}.model --lang ${tgt_lang} \
            data/${ttask} ${dict} > ${feat_trans_dir}/data_${bpemode}${nbpe}.${tgt_case}.json
    done
    echo "update json (add source references)"
    # update json (add source references)
    for x in ${train_set} ${train_dev}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data/$(echo ${x} | cut -f 1 -d ".").en-${tgt_lang}.en
        update_json.sh --text ${data_dir}/text.${src_case} --bpecode ${bpemodel}.model \
            ${feat_dir}/data_${bpemode}${nbpe}.${tgt_case}.json ${data_dir} ${dict}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    x=(${train_set} ${train_dev} ${trans_set})
    y=(train dev test)
    for (( i=0; i<${#x[*]}; ++i)); do
        echo ${x[$i]} ${y[$i]}
        feat_dir=${dumpdir}/${x[$i]}/delta${do_delta}
        data_dir=data/$(echo ${x[$i]} | cut -f 1 -d ".").en-${tgt_lang}.en
        python3 ${MAIN_ROOT}/utils/espnet_json_to_manifest.py \
                --json-file ${feat_dir}/data_${bpemode}${nbpe}.${tgt_case}.json \
                --manifest-file data/manifest.${tgt_lang}.${y[$i]}
        echo "Process done for ${y[$i]} set from ${x[$i]}"
    done
fi


echo "MuST-C ${tgt_lang} Data preparation done."
exit 0
