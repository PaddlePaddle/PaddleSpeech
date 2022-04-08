#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

stage=0
stop_stage=0

# TODO: tacotron2 动转静的结果没有动态图的响亮, 可能还是 decode 的时候某个函数动静不对齐
# pwgan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=tacotron2_csmsc \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=pwgan_csmsc \
        --voc_config=pwg_baker_ckpt_0.4/pwg_default.yaml \
        --voc_ckpt=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
        --voc_stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --inference_dir=${train_output_path}/inference
        
fi

# for more GAN Vocoders
# multi band melgan
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=tacotron2_csmsc \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=mb_melgan_csmsc \
        --voc_config=mb_melgan_csmsc_ckpt_0.1.1/default.yaml \
        --voc_ckpt=mb_melgan_csmsc_ckpt_0.1.1/snapshot_iter_1000000.pdz\
        --voc_stat=mb_melgan_csmsc_ckpt_0.1.1/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --inference_dir=${train_output_path}/inference
fi

# the pretrained models haven't release now
# style melgan
# style melgan's Dygraph to Static Graph is not ready now
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=tacotron2_csmsc \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=style_melgan_csmsc \
        --voc_config=style_melgan_csmsc_ckpt_0.1.1/default.yaml \
        --voc_ckpt=style_melgan_csmsc_ckpt_0.1.1/snapshot_iter_1500000.pdz \
        --voc_stat=style_melgan_csmsc_ckpt_0.1.1/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt
        # --inference_dir=${train_output_path}/inference
fi

# hifigan
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "in hifigan syn_e2e"
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=tacotron2_csmsc \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_csmsc \
        --voc_config=hifigan_csmsc_ckpt_0.1.1/default.yaml \
        --voc_ckpt=hifigan_csmsc_ckpt_0.1.1/snapshot_iter_2500000.pdz \
        --voc_stat=hifigan_csmsc_ckpt_0.1.1/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --inference_dir=${train_output_path}/inference
fi

# wavernn
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "in wavernn syn_e2e"
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=tacotron2_csmsc \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=wavernn_csmsc \
        --voc_config=wavernn_csmsc_ckpt_0.2.0/default.yaml \
        --voc_ckpt=wavernn_csmsc_ckpt_0.2.0/snapshot_iter_400000.pdz \
        --voc_stat=wavernn_csmsc_ckpt_0.2.0/feats_stats.npy \
        --lang=zh \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --inference_dir=${train_output_path}/inference
fi
