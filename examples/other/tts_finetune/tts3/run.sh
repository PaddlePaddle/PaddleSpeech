#!/bin/bash

set -e
source path.sh


input_dir=./input/csmsc_mini
newdir_name="newdir"
new_dir=${input_dir}/${newdir_name}
pretrained_model_dir=./pretrained_models/fastspeech2_aishell3_ckpt_1.1.0
mfa_dir=./mfa_result
dump_dir=./dump
output_dir=./exp/default
lang=zh
ngpu=1
finetune_config=./conf/finetune.yaml
replace_spkid=0

ckpt=snapshot_iter_96699

gpus=1
CUDA_VISIBLE_DEVICES=${gpus}
stage=0
stop_stage=100


# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

# check oov
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "check oov"
    python3 local/check_oov.py \
        --input_dir=${input_dir} \
        --pretrained_model_dir=${pretrained_model_dir} \
        --newdir_name=${newdir_name} \
        --lang=${lang}
fi

# get mfa result
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "get mfa result"
    python3 local/get_mfa_result.py \
        --input_dir=${new_dir} \
        --mfa_dir=${mfa_dir} \
        --lang=${lang}
fi

# generate durations.txt
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "generate durations.txt"
    python3 local/generate_duration.py \
        --mfa_dir=${mfa_dir} 
fi

# extract feature
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "extract feature"
    python3 local/extract_feature.py \
        --duration_file="./durations.txt" \
        --input_dir=${new_dir} \
        --dump_dir=${dump_dir} \
        --pretrained_model_dir=${pretrained_model_dir} \
        --replace_spkid=$replace_spkid
fi

# create finetune env
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "create finetune env"
    python3 local/prepare_env.py \
        --pretrained_model_dir=${pretrained_model_dir} \
        --output_dir=${output_dir}
fi

# finetune
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "finetune..."
    python3 local/finetune.py \
        --pretrained_model_dir=${pretrained_model_dir} \
        --dump_dir=${dump_dir} \
        --output_dir=${output_dir} \
        --ngpu=${ngpu} \
        --epoch=100 \
        --finetune_config=${finetune_config}
fi

# synthesize e2e
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "in hifigan syn_e2e"
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=fastspeech2_aishell3 \
        --am_config=${pretrained_model_dir}/default.yaml \
        --am_ckpt=${output_dir}/checkpoints/${ckpt}.pdz \
        --am_stat=${pretrained_model_dir}/speech_stats.npy \
        --voc=hifigan_aishell3 \
        --voc_config=pretrained_models/hifigan_aishell3_ckpt_0.2.0/default.yaml \
        --voc_ckpt=pretrained_models/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz \
        --voc_stat=pretrained_models/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../../assets/sentences.txt \
        --output_dir=./test_e2e/ \
        --phones_dict=${dump_dir}/phone_id_map.txt \
        --speaker_dict=${dump_dir}/speaker_id_map.txt \
        --spk_id=$replace_spkid
fi
