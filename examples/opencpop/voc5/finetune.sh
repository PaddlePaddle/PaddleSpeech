#!/bin/bash

source path.sh

gpus=0
stage=0
stop_stage=100

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${MAIN_ROOT}/paddlespeech/t2s/exps/diffsinger/gen_gta_mel.py \
        --diffsinger-config=diffsinger_opencpop_ckpt_1.4.0/default.yaml \
        --diffsinger-checkpoint=diffsinger_opencpop_ckpt_1.4.0/snapshot_iter_160000.pdz \
        --diffsinger-stat=diffsinger_opencpop_ckpt_1.4.0/speech_stats.npy \
        --diffsinger-stretch=diffsinger_opencpop_ckpt_1.4.0/speech_stretchs.npy \
        --dur-file=~/datasets/Opencpop/segments/transcriptions.txt \
        --output-dir=dump_finetune \
        --phones-dict=diffsinger_opencpop_ckpt_1.4.0/phone_id_map.txt \
        --dataset=opencpop \
        --rootdir=~/datasets/Opencpop/segments/
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${MAIN_ROOT}/utils/link_wav.py \
        --old-dump-dir=dump \
        --dump-dir=dump_finetune
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # get features' stats(mean and std)
    echo "Get features' stats ..."
    cp dump/train/feats_stats.npy dump_finetune/train/
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # normalize, dev and test should use train's stats
    echo "Normalize ..."
   
    python3 ${BIN_DIR}/../normalize.py \
        --metadata=dump_finetune/train/raw/metadata.jsonl \
        --dumpdir=dump_finetune/train/norm \
        --stats=dump_finetune/train/feats_stats.npy
    python3 ${BIN_DIR}/../normalize.py \
        --metadata=dump_finetune/dev/raw/metadata.jsonl \
        --dumpdir=dump_finetune/dev/norm \
        --stats=dump_finetune/train/feats_stats.npy
    
    python3 ${BIN_DIR}/../normalize.py \
        --metadata=dump_finetune/test/raw/metadata.jsonl \
        --dumpdir=dump_finetune/test/norm \
        --stats=dump_finetune/train/feats_stats.npy
fi

# create finetune env
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "create finetune env"
    python3 local/prepare_env.py \
        --pretrained_model_dir=exp/default/checkpoints/ \
        --output_dir=exp/finetune/
fi 

# finetune
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} \
    FLAGS_cudnn_exhaustive_search=true \
    FLAGS_conv_workspace_size_limit=4000 \
    python ${BIN_DIR}/train.py \
        --train-metadata=dump_finetune/train/norm/metadata.jsonl \
        --dev-metadata=dump_finetune/dev/norm/metadata.jsonl \
        --config=conf/finetune.yaml \
        --output-dir=exp/finetune \
        --ngpu=1
fi 
