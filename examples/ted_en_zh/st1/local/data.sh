#!/bin/bash

set -e

stage=-1
stop_stage=100
dict_dir=data/lang_char

# bpemode (unigram or bpe)
nbpe=8000
bpemode=bpe
bpeprefix="${dict_dir}/bpe_${bpemode}_${nbpe}"
data_dir=./TED_EnZh
target_dir=data/ted_en_zh
dumpdir=data/dump
do_delta=false
nj=20

source ${MAIN_ROOT}/utils/parse_options.sh

TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}
mkdir -p data
mkdir -p ${dict_dir}


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ ! -e ${data_dir} ]; then
        echo "Error: Dataset is not avaiable. Please download and unzip the dataset"
        echo "Download Link: https://pan.baidu.com/s/18L-59wgeS96WkObISrytQQ Passwd: bva0"
        echo "The tree of the directory should be:"
        echo "."
        echo "|-- En-Zh"
        echo "|-- test-segment"
        echo "    |-- tst2010"
        echo "    |-- ..."
        echo "|-- train-split"
        echo "    |-- train-segment"
        echo "|-- README.md"

        exit 1
    fi

    # extract data 
    echo "data Extraction"
    python3 local/ted_en_zh.py \
    --tgt-dir=${target_dir} \
    --src-dir=${data_dir}

fi
prep_dir=${target_dir}/data_prep 
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
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

        cat ${dst}/utt2spk | utils/utt2spk_to_spk2utt.pl | sort -k1,1 -u > ${dst}/spk2utt
        rm -rf ${prep_dir}/${set}.en-zh
        mkdir -p ${prep_dir}/${set}.en-zh
        echo "remove duplicate lines..."
        cut -d ' ' -f 1 ${dst}/text.en | sort | uniq -c | sort -n -k1 -r | grep -v '1 ted-en-zh' \
            | sed 's/^[ \t]*//' > ${dst}/duplicate_lines
        cut -d ' ' -f 1 ${dst}/text.en | sort | uniq -c | sort -n -k1 -r | grep '1 ted-en-zh' \
            | cut -d '1' -f 2- | sed 's/^[ \t]*//' > ${dst}/reclist
        reduce_data_dir.sh ${dst} ${dst}/reclist ${prep_dir}/${set}.en-zh
        echo "done wav processing"
        for l in en zh; do
            cp ${dst}/text.${l} ${prep_dir}/${set}.en-zh/text.${l}
        done
        utils/fix_data_dir.sh --utt_extra_files \
        "text.en text.zh" \
        ${prep_dir}/${set}.en-zh
    done
fi

feat_tr_dir=${dumpdir}/train_sp/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/dev/delta${do_delta}; mkdir -p ${feat_dt_dir}
feat_trans_dir=${dumpdir}/test/delta${do_delta}; mkdir -p ${feat_trans_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=data/fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train dev test; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            ${prep_dir}/${x}.en-zh data/make_fbank/${x} ${fbankdir}
    done
    
    echo "speed perturbation"
    utils/perturb_data_dir_speed.sh 0.9 ${prep_dir}/train.en-zh ${prep_dir}/temp1.en-zh
    utils/perturb_data_dir_speed.sh 1.0 ${prep_dir}/train.en-zh ${prep_dir}/temp2.en-zh
    utils/perturb_data_dir_speed.sh 1.1 ${prep_dir}/train.en-zh ${prep_dir}/temp3.en-zh

    utils/combine_data.sh --extra-files utt2uniq ${prep_dir}/train_sp.en-zh \
    ${prep_dir}/temp1.en-zh ${prep_dir}/temp2.en-zh ${prep_dir}/temp3.en-zh
    rm -r ${prep_dir}/temp*.en-zh 
    utils/fix_data_dir.sh ${prep_dir}/train_sp.en-zh

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
        ${prep_dir}/train_sp.en-zh exp/make_fbank/train_sp.en-zh ${fbankdir}

    for lang in en zh; do
        cat /dev/null > ${prep_dir}/train_sp.en-zh/text.${lang}
        for p in "sp0.9-" "sp1.0-" "sp1.1-"; do
            awk -v p=${p} '{printf("%s %s%s\n", $1, p, $1);}' ${prep_dir}/train.en-zh/utt2spk > ${prep_dir}/train_sp.en-zh/utt_map
            utils/apply_map.pl -f 1 ${prep_dir}/train_sp.en-zh/utt_map < ${prep_dir}/train.en-zh/text.${lang} >>${prep_dir}/train_sp.en-zh/text.${lang}
        done
    done

    for x in train_sp dev test; do
        local/divide_lang.sh ${prep_dir}/${x}.en-zh zh
    done

    for x in train_sp dev; do
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        for lang in zh en; do
            remove_longshortdata.sh --maxframes 3000 --maxchars 400 ${prep_dir}/${x}.en-zh.${lang} ${prep_dir}/${x}.en-zh.${lang}.tmp
        done
        cut -f 1 -d " " ${prep_dir}/${x}.en-zh.en.tmp/text > ${prep_dir}/${x}.en-zh.${lang}.tmp/reclist1
        cut -f 1 -d " " ${prep_dir}/${x}.en-zh.${lang}.tmp/text > ${prep_dir}/${x}.en-zh.${lang}.tmp/reclist2
        comm -12 ${prep_dir}/${x}.en-zh.${lang}.tmp/reclist1 ${prep_dir}/${x}.en-zh.${lang}.tmp/reclist2 > ${prep_dir}/${x}.en-zh.en.tmp/reclist

        for lang in zh en; do
            reduce_data_dir.sh ${prep_dir}/${x}.en-zh.${lang}.tmp ${prep_dir}/${x}.en-zh.en.tmp/reclist ${prep_dir}/${x}.en-zh.${lang}
            utils/fix_data_dir.sh  ${prep_dir}/${x}.en-zh.${lang}
        done
        rm -rf ${prep_dir}/${x}.en-zh.*.tmp
    done

    compute-cmvn-stats scp:${prep_dir}/train_sp.en-zh.zh/feats.scp ${prep_dir}/train_sp.en-zh.zh/cmvn.ark

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta $do_delta \
        ${prep_dir}/train_sp.en-zh.zh/feats.scp ${prep_dir}/train_sp.en-zh.zh/cmvn.ark ${prep_dir}/dump_feats/train_sp.en-zh.zh ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta $do_delta \
        ${prep_dir}/dev.en-zh.zh/feats.scp ${prep_dir}/train_sp.en-zh.zh/cmvn.ark ${prep_dir}/dump_feats/dev.en-zh.zh ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta $do_delta \
        ${prep_dir}/test.en-zh.zh/feats.scp ${prep_dir}/train_sp.en-zh.zh/cmvn.ark ${prep_dir}/dump_feats/test.en-zh.zh ${feat_trans_dir}
fi

dict=${dict_dir}/ted_en_zh_${bpemode}${nbpe}.txt
nlsyms=${dict_dir}/ted_en_zh_non_lang_syms.txt
bpemodel=${dict_dir}/ted_en_zh_${bpemode}${nbpe}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"

    echo "make a joint source and target dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    offset=$(wc -l < ${dict})
    grep sp1.0 ${prep_dir}/train_sp.en-zh.*/text | cut -f 2- -d' ' | grep -v -e '^\s*$' > ${dict_dir}/input.txt
    spm_train  --input=${dict_dir}/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${bpemodel}.model --output_format=piece < ${dict_dir}/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --nj ${nj} --feat ${feat_tr_dir}/feats.scp --text ${prep_dir}/train_sp.en-zh.zh/text --bpecode ${bpemodel}.model --lang zh \
        ${prep_dir}/train_sp.en-zh.zh ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --text ${prep_dir}/dev.en-zh.zh/text --bpecode ${bpemodel}.model --lang zh \
        ${prep_dir}/dev.en-zh.zh ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_trans_dir}/feats.scp --text ${prep_dir}/test.en-zh.zh/text --bpecode ${bpemodel}.model --lang zh \
        ${prep_dir}/test.en-zh.zh ${dict} > ${feat_trans_dir}/data_${bpemode}${nbpe}.json
    echo "update json (add source references)"
    # update json (add source references)
    for x in train_sp dev; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=${prep_dir}/$(echo ${x} | cut -f 1 -d ".").en-zh.en
        update_json.sh --text ${data_dir}/text --bpecode ${bpemodel}.model \
            ${feat_dir}/data_${bpemode}${nbpe}.json ${data_dir} ${dict}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    x=(${feat_tr_dir} ${feat_dt_dir} ${feat_trans_dir})
    y=(train dev test)
    echo "stage 3: Format the Json Data"
    for (( i=0; i<${#x[*]}; ++i)); do
        python3 ${MAIN_ROOT}/utils/espnet_json_to_manifest.py \
         --json-file ${x[$i]}/data_${bpemode}${nbpe}.json \
         --manifest-file data/manifest.${y[$i]}
    done
fi
echo "Ted En-Zh Data preparation done."
exit 0
