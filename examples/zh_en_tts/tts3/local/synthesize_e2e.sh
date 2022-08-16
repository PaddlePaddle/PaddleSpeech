#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

stage=0
stop_stage=0

# voc: pwgan_aishell3
# the spk_id=174 means baker speaker, default.
# the spk_id=175 means ljspeech speaker
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=fastspeech2_mix \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=pwgan_aishell3 \
        --voc_config=pwg_aishell3_ckpt_0.5/default.yaml \
        --voc_ckpt=pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
        --voc_stat=pwg_aishell3_ckpt_0.5/feats_stats.npy \
        --lang=mix \
        --text=${BIN_DIR}/../sentences_mix.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --spk_id=174 \
        --inference_dir=${train_output_path}/inference
fi

# voc: hifigan_aishell3
# the spk_id=174 means baker speaker, default
# the spk_id=175 means ljspeech speaker
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "in hifigan syn_e2e"
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=fastspeech2_mix \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_aishell3 \
        --voc_config=hifigan_aishell3_ckpt_0.2.0/default.yaml \
        --voc_ckpt=hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz \
        --voc_stat=hifigan_aishell3_ckpt_0.2.0/feats_stats.npy \
        --lang=mix \
        --text=${BIN_DIR}/../sentences_mix.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --spk_id=174 \
        --inference_dir=${train_output_path}/inference
fi


# voc: hifigan_csmsc
# when speaker is 174 (csmsc), use csmsc's vocoder is better than aishell3's
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "in csmsc's hifigan syn_e2e"
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=fastspeech2_mix \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_csmsc \
        --voc_config=hifigan_csmsc_ckpt_0.1.1/default.yaml \
        --voc_ckpt=hifigan_csmsc_ckpt_0.1.1/snapshot_iter_2500000.pdz \
        --voc_stat=hifigan_csmsc_ckpt_0.1.1/feats_stats.npy \
        --lang=mix \
        --text=${BIN_DIR}/../sentences_mix.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --spk_id=174 \
        --inference_dir=${train_output_path}/inference
fi