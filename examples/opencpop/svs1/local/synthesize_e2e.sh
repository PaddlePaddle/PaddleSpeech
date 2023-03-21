#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

stage=0
stop_stage=0

# pwgan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=diffsinger_opencpop \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=pwgan_opencpop \
        --voc_config=pwgan_opencpop_ckpt_1.4.0/default.yaml \
        --voc_ckpt=pwgan_opencpop_ckpt_1.4.0/snapshot_iter_100000.pdz \
        --voc_stat=pwgan_opencpop_ckpt_1.4.0/feats_stats.npy \
        --lang=sing \
        --text=${BIN_DIR}/../sentences_sing.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --speech_stretchs=dump/train/speech_stretchs.npy \
        --pinyin_phone=local/pinyin_to_phone.txt
fi

# for more GAN Vocoders
# hifigan
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "in hifigan syn_e2e"
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=diffsinger_opencpop \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_opencpop \
        --voc_config=hifigan_opencpop_ckpt_1.4.0/default.yaml \
        --voc_ckpt=hifigan_opencpop_ckpt_1.4.0/snapshot_iter_625000.pdz \
        --voc_stat=hifigan_opencpop_ckpt_1.4.0/feats_stats.npy \
        --lang=sing \
        --text=${BIN_DIR}/../sentences_sing.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --speech_stretchs=dump/train/speech_stretchs.npy \
        --pinyin_phone=local/pinyin_to_phone.txt
        
fi
