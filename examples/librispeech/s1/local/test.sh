#!/bin/bash

set -e

expdir=exp
datadir=data
nj=32

lmtag=

recog_set="test-clean test-other dev-clean dev-other"
recog_set="test-clean"

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
bpeprefix="data/bpe_${bpemode}_${nbpe}"
bpemodel=${bpeprefix}.model

if [ $# != 3 ];then
    echo "usage: ${0} config_path dict_path ckpt_path_prefix"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
dict=$2
ckpt_prefix=$3

chunk_mode=false
if [[ ${config_path} =~ ^.*chunk_.*yaml$ ]];then
    chunk_mode=true
fi
echo "chunk mode ${chunk_mode}"


# download language model
#bash local/download_lm_en.sh
#if [ $? -ne 0 ]; then
#    exit 1
#fi

pids=() # initialize pids

for dmethd in attention ctc_greedy_search ctc_prefix_beam_search attention_rescoring; do
(
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${dmethd}_$(basename ${config_path%.*})_${lmtag}
        feat_recog_dir=${datadir}
        mkdir -p ${expdir}/${decode_dir}
        mkdir -p ${feat_recog_dir}

        # split data
        split_json.sh ${feat_recog_dir}/manifest.${rtask} ${nj}

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        batch_size=1
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            python3 -u ${BIN_DIR}/test.py \
            --nproc ${ngpu} \
            --config ${config_path} \
            --result_file ${expdir}/${decode_dir}/data.JOB.json \
            --checkpoint_path ${ckpt_prefix} \
            --opts decoding.decoding_method ${dmethd} \
            --opts decoding.batch_size ${batch_size} \
            --opts data.test_manifest ${feat_recog_dir}/split${nj}/JOB/manifest.${rtask}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
) &
pids+=($!) # store background pids
done

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished"

exit 0
