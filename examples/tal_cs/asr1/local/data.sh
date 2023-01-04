#!/bin/bash
stage=-1
stop_stage=100
dict_dir=data/lang_char

# bpemode (unigram or bpe)
nbpe=11297
bpemode=bpe
bpeprefix="${dict_dir}/bpe_${bpemode}_${nbpe}"

stride_ms=10
window_ms=25
sample_rate=16000
feat_dim=80

source ${MAIN_ROOT}/utils/parse_options.sh


mkdir -p data
mkdir -p ${dict_dir}
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}

#prepare data
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ ! -d "${MAIN_ROOT}/dataset/tal_cs/TALCS_corpus" ]; then
        echo "${MAIN_ROOT}/dataset/tal_cs/TALCS_corpus does not exist. Please donwload tal_cs data and unpack it from https://ai.100tal.com/dataset first."
        echo "data md5 reference: 4c879b3c9c05365fc9dee1fc68713afe"
        exit
    fi
    # create manifest json file from TALCS_corpus
    python ${MAIN_ROOT}/dataset/tal_cs/tal_cs.py --target_dir ${MAIN_ROOT}/dataset/tal_cs/TALCS_corpus/ --manifest_prefix data/
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # compute mean and stddev for normalizer
    num_workers=$(nproc)
    python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
    --manifest_path="data/manifest.train.raw" \
    --num_samples=-1 \
    --spectrum_type="fbank" \
    --feat_dim=${feat_dim}  \
    --delta_delta=false \
    --sample_rate=${sample_rate} \
    --stride_ms=${stride_ms} \
    --window_ms=${window_ms} \
    --use_dB_normalization=False \
    --num_workers=${num_workers} \
    --output_path="data/mean_std.json"
    echo "compute mean and stddev done."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    #use train_set build dict
    python3 ${MAIN_ROOT}/utils/build_vocab.py \
    --unit_type 'spm' \
    --count_threshold=0 \
    --vocab_path="${dict_dir}/vocab.txt"  \
    --manifest_paths="data/manifest.train.raw"  \
    --spm_mode=${bpemode} \
    --spm_vocab_size=${nbpe}  \
    --spm_model_prefix=${bpeprefix} \
    --spm_character_coverage=1 
    echo "build dict done."
fi

#use new dict format data
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # format manifest with tokenids, vocab size
    for sub in train dev test ; do
    {
        python3 ${MAIN_ROOT}/utils/format_data.py \
        --cmvn_path "data/mean_std.json" \
        --unit_type "spm" \
        --spm_model_prefix ${bpeprefix} \
        --vocab_path="${dict_dir}/vocab.txt" \
        --manifest_path="data/manifest.${sub}.raw" \
        --output_path="data/manifest.${sub}"

        if [ $? -ne 0 ]; then
            echo "Formt mnaifest failed. Terminated."
            exit 1
        fi
    }&
    done
    wait
    echo "format data done."
fi
