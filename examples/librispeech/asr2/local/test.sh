#!/bin/bash

set -e

expdir=exp
datadir=data
nj=32

lmtag='nolm'

train_set=train_960
recog_set="test-clean test-other dev-clean dev-other"
recog_set="test-clean"

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
bpeprefix=data/lang_char/${train_set}_${bpemode}${nbpe}
bpemodel=${bpeprefix}.model

config_path=conf/transformer.yaml
decode_config_path=conf/decode/decode_base.yaml
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
ckpt_prefix=

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

if [ -z ${ckpt_prefix} ]; then
    echo "usage: $0 --ckpt_prefix ckpt_prefix"
    exit 1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

ckpt_dir=$(dirname `dirname ${ckpt_prefix}`)
echo "ckpt dir: ${ckpt_dir}"

ckpt_tag=$(basename ${ckpt_prefix})
echo "ckpt tag: ${ckpt_tag}"

chunk_mode=false
if [[ ${config_path} =~ ^.*chunk_.*yaml$ ]];then
    chunk_mode=true
fi
echo "chunk mode: ${chunk_mode}"


# download language model
#bash local/download_lm_en.sh
#if [ $? -ne 0 ]; then
#    exit 1
#fi

pids=() # initialize pids

for dmethd in attention ctc_greedy_search ctc_prefix_beam_search attention_rescoring; do
(
    echo "decode method: ${dmethd}"
    for rtask in ${recog_set}; do
    (
        echo "dataset: ${rtask}"
        decode_dir=${ckpt_dir}/decode/decode_${rtask/-/_}_${dmethd}_$(basename ${config_path%.*})_${lmtag}_${ckpt_tag}
        feat_recog_dir=${datadir}
        mkdir -p ${decode_dir}
        mkdir -p ${feat_recog_dir}

        # split data
        split_json.sh manifest.${rtask} ${nj}

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        batch_size=1
        ${decode_cmd} JOB=1:${nj} ${decode_dir}/log/decode.JOB.log \
            python3 -u ${BIN_DIR}/test.py \
            --model-name u2_kaldi \
            --run-mode test \
            --ngpu ${ngpu} \
            --dict-path ${dict} \
            --config ${config_path} \
            --decode_cfg ${decode_config_path} \
            --checkpoint_path ${ckpt_prefix} \
            --result-file ${decode_dir}/data.JOB.json \
            --opts decode.decoding_method ${dmethd} \
            --opts decode.decode_batch_size ${batch_size} \
            --opts test_manifest ${feat_recog_dir}/split${nj}/JOB/manifest.${rtask}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel} --wer false ${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." || true
    done
)
done

echo "Finished"

exit 0
