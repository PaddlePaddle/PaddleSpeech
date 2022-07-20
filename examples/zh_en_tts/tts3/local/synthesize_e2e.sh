#!/bin/bash

model_dir=$1
output=$2
am_name=fastspeech2_csmscljspeech_add-zhen
am_model_dir=${model_dir}/${am_name}/

stage=1
stop_stage=1


# hifigan
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=fastspeech2_mix \
        --am_config=${am_model_dir}/default.yaml \
        --am_ckpt=${am_model_dir}/snapshot_iter_94000.pdz \
        --am_stat=${am_model_dir}/speech_stats.npy \
        --voc=hifigan_ljspeech \
        --voc_config=${model_dir}/hifigan_ljspeech_ckpt_0.2.0/default.yaml \
        --voc_ckpt=${model_dir}/hifigan_ljspeech_ckpt_0.2.0/snapshot_iter_2500000.pdz \
        --voc_stat=${model_dir}/hifigan_ljspeech_ckpt_0.2.0/feats_stats.npy \
        --lang=mix \
        --text=${BIN_DIR}/../sentences_mix.txt \
        --output_dir=${output}/test_e2e \
        --phones_dict=${am_model_dir}/phone_id_map.txt \
        --speaker_dict=${am_model_dir}/speaker_id_map.txt \
        --spk_id 0 
fi
