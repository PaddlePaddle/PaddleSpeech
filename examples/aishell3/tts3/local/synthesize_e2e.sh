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
        --am=fastspeech2_aishell3 \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=pwgan_aishell3 \
        --voc_config=pwg_aishell3_ckpt_0.5/default.yaml \
        --voc_ckpt=pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
        --voc_stat=pwg_aishell3_ckpt_0.5/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --spk_id=0 \
        --inference_dir=${train_output_path}/inference
fi

# hifigan
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "in hifigan syn_e2e"
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=fastspeech2_aishell3 \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=fastspeech2_nosil_aishell3_ckpt_0.4/speech_stats.npy \
        --voc=hifigan_aishell3 \
        --voc_config=hifigan_aishell3_ckpt_0.2.0/default.yaml \
        --voc_ckpt=hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz \
        --voc_stat=hifigan_aishell3_ckpt_0.2.0/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=fastspeech2_nosil_aishell3_ckpt_0.4/phone_id_map.txt \
        --speaker_dict=fastspeech2_nosil_aishell3_ckpt_0.4/speaker_id_map.txt \
        --spk_id=0 \
        --inference_dir=${train_output_path}/inference
    fi
