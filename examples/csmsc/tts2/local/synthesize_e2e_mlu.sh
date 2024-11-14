#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

stage=0
stop_stage=0

# for more GAN Vocoders
# multi band melgan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=speedyspeech_csmsc \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/feats_stats.npy \
        --voc=mb_melgan_csmsc \
        --voc_config=mb_melgan_csmsc_ckpt_0.1.1/default.yaml \
        --voc_ckpt=mb_melgan_csmsc_ckpt_0.1.1/snapshot_iter_1000000.pdz\
        --voc_stat=mb_melgan_csmsc_ckpt_0.1.1/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../../assets/sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --tones_dict=dump/tone_id_map.txt \
        --ngpu=0 \
        --nmlu=1
fi

# the pretrained models haven't release now
# style melgan
# style melgan's Dygraph to Static Graph is not ready now
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=speedyspeech_csmsc \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/feats_stats.npy \
        --voc=style_melgan_csmsc \
        --voc_config=style_melgan_csmsc_ckpt_0.1.1/default.yaml \
        --voc_ckpt=style_melgan_csmsc_ckpt_0.1.1/snapshot_iter_1500000.pdz \
        --voc_stat=style_melgan_csmsc_ckpt_0.1.1/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../../assets/sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --tones_dict=dump/tone_id_map.txt \
        --ngpu=0 \
        --nmlu=1
        # --inference_dir=${train_output_path}/inference
fi

# hifigan
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=speedyspeech_csmsc \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/feats_stats.npy \
        --voc=hifigan_csmsc \
        --voc_config=hifigan_csmsc_ckpt_0.1.1/default.yaml \
        --voc_ckpt=hifigan_csmsc_ckpt_0.1.1/snapshot_iter_2500000.pdz \
        --voc_stat=hifigan_csmsc_ckpt_0.1.1/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../../assets/sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --tones_dict=dump/tone_id_map.txt \
        --inference_dir=${train_output_path}/inference \
        --ngpu=0 \
        --nmlu=1
fi

# wavernn
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "in wavernn syn_e2e"
    FLAGS_allocator_strategy=naive_best_fit \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=speedyspeech_csmsc \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/feats_stats.npy \
        --voc=wavernn_csmsc \
        --voc_config=wavernn_csmsc_ckpt_0.2.0/default.yaml \
        --voc_ckpt=wavernn_csmsc_ckpt_0.2.0/snapshot_iter_400000.pdz \
        --voc_stat=wavernn_csmsc_ckpt_0.2.0/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../../assets/sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --tones_dict=dump/tone_id_map.txt \
        --inference_dir=${train_output_path}/inference \
        --ngpu=0 \
        --nmlu=1
fi
