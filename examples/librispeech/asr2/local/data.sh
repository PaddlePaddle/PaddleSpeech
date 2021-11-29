#!/bin/bash

stage=-1
stop_stage=100
nj=32
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=${MAIN_ROOT}/dataset/

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

source ${MAIN_ROOT}/utils/parse_options.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_sp=train_sp
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"


mkdir -p data
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data, generate manifests
    python3 ${TARGET_DIR}/librispeech/librispeech.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${TARGET_DIR}/librispeech" \
    --full_download="True"

    if [ $? -ne 0 ]; then
        echo "Prepare LibriSpeech failed. Terminated."
        exit 1
    fi

    for set in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
        mv data/manifest.${set} data/manifest.${set}.raw
    done

    rm -rf data/manifest.train.raw data/manifest.dev.raw  data/manifest.test.raw
    for set in train-clean-100 train-clean-360 train-other-500; do
        cat data/manifest.${set}.raw >> data/manifest.train.raw
    done

    for set in dev-clean dev-other; do
        cat data/manifest.${set}.raw >> data/manifest.dev.raw
    done

    for set in test-clean test-other; do
        cat data/manifest.${set}.raw >> data/manifest.test.raw
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/librispeech/${part}/LibriSpeech/${part} data/${part//-/_}
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_sp_dir=${dumpdir}/${train_sp}/delta${do_delta}; mkdir -p ${feat_sp_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/combine_data.sh --extra_files utt2num_frames data/${train_set}_org data/train_clean_100 data/train_clean_360 data/train_other_500
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev}_org data/dev_clean data/dev_other
    utils/perturb_data_dir_speed.sh 0.9  data/${train_set}_org  data/temp1
    utils/perturb_data_dir_speed.sh 1.0  data/${train_set}_org  data/temp2
    utils/perturb_data_dir_speed.sh 1.1  data/${train_set}_org  data/temp3

    utils/combine_data.sh --extra-files utt2uniq data/${train_sp}_org data/temp1 data/temp2 data/temp3

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_sp}_org data/${train_sp}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj $nj  --write_utt2num_frames true \
            data/train_sp  exp/make_fbank/train_sp  ${fbankdir}
    utils/fix_data_dir.sh data/train_sp
    # compute global CMVN
    compute-cmvn-stats scp:data/${train_sp}/feats.scp data/${train_sp}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_sp}/feats.scp data/${train_sp}/cmvn.ark exp/dump_feats/train ${feat_sp_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_sp}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_sp}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --nj ${nj} --feat ${feat_sp_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_sp} ${dict} > ${feat_sp_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --nj ${nj} --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --nj ${nj} --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # make json labels
    python3 local/espnet_json_to_manifest.py --json-file ${feat_sp_dir}/data_${bpemode}${nbpe}.json --manifest-file data/manifest.train
    python3 local/espnet_json_to_manifest.py --json-file ${feat_dt_dir}/data_${bpemode}${nbpe}.json --manifest-file data/manifest.dev

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        python3 local/espnet_json_to_manifest.py --json-file ${feat_recog_dir}/data_${bpemode}${nbpe}.json --manifest-file data/manifest.${rtask//_/-}
    done
fi

echo "LibriSpeech Data preparation done."
exit 0
