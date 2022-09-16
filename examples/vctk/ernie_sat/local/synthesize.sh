#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

stage=1
stop_stage=1

# use am to predict duration here
# 增加 am_phones_dict am_tones_dict 等，也可以用新的方式构造 am, 不需要这么多参数了就

# pwgan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/synthesize.py \
        --erniesat_config=${config_path} \
        --erniesat_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --erniesat_stat=dump/train/speech_stats.npy \
        --voc=pwgan_vctk \
        --voc_config=pwg_vctk_ckpt_0.1.1/default.yaml  \
        --voc_ckpt=pwg_vctk_ckpt_0.1.1/snapshot_iter_1500000.pdz \
        --voc_stat=pwg_vctk_ckpt_0.1.1/feats_stats.npy \
        --test_metadata=dump/test/norm/metadata.jsonl \
        --output_dir=${train_output_path}/test \
        --phones_dict=dump/phone_id_map.txt
fi

# hifigan
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/synthesize.py \
        --erniesat_config=${config_path} \
        --erniesat_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --erniesat_stat=dump/train/speech_stats.npy \
        --voc=hifigan_vctk \
        --voc_config=hifigan_vctk_ckpt_0.2.0/default.yaml  \
        --voc_ckpt=hifigan_vctk_ckpt_0.2.0/snapshot_iter_2500000.pdz \
        --voc_stat=hifigan_vctk_ckpt_0.2.0/feats_stats.npy \
        --test_metadata=dump/test/norm/metadata.jsonl \
        --output_dir=${train_output_path}/test \
        --phones_dict=dump/phone_id_map.txt
fi
