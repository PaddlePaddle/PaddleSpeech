#!/bin/bash

set -e
source path.sh


input_dir=./input/baker_mini
pretrained_model_dir=./pretrained_models/fastspeech2_aishell3_ckpt_1.1.0
mfa_dir=./mfa_result
dump_dir=./dump
output_dir=./exp/default
lang=zh
ngpu=2

ckpt=snapshot_iter_96600

gpus=0,1
CUDA_VISIBLE_DEVICES=${gpus}
stage=0
stop_stage=100


# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # finetune
    python3 finetune.py \
        --input_dir=${input_dir} \
        --pretrained_model_dir=${pretrained_model_dir} \
        --mfa_dir=${mfa_dir} \
        --dump_dir=${dump_dir} \
        --output_dir=${output_dir} \
        --lang=${lang} \
        --ngpu=${ngpu} \
        --epoch=100
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "in hifigan syn_e2e"
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
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
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=./test_e2e \
        --phones_dict=${dump_dir}/phone_id_map.txt \
        --speaker_dict=${dump_dir}/speaker_id_map.txt \
        --spk_id=0 
fi
