#!/bin/bash

set -e
source path.sh

# 纯英文的语音编辑
# 样例为把 p243_new 对应的原始语音: For that reason cover should not be given.编辑成 'for that reason cover is impossible to be given.' 对应的语音
# NOTE: 语音编辑任务暂支持句子中 1 个位置的替换或者插入文本操作

python local/inference_new.py \
    --task_name=edit \
    --model_name=paddle_checkpoint_en \
    --uid=p243_new \
    --new_str='for that reason cover is impossible to be given.' \
    --prefix='./prompt/dev/' \
    --source_lang=english \
    --target_lang=english \
    --output_name=pred_edit.wav \
    --voc=pwgan_aishell3 \
    --voc_config=download/pwg_aishell3_ckpt_0.5/default.yaml \
    --voc_ckpt=download/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
    --voc_stat=download/pwg_aishell3_ckpt_0.5/feats_stats.npy \
    --am=fastspeech2_ljspeech \
    --am_config=download/fastspeech2_nosil_ljspeech_ckpt_0.5/default.yaml \
    --am_ckpt=download/fastspeech2_nosil_ljspeech_ckpt_0.5/snapshot_iter_100000.pdz \
    --am_stat=download/fastspeech2_nosil_ljspeech_ckpt_0.5/speech_stats.npy \
    --phones_dict=download/fastspeech2_nosil_ljspeech_ckpt_0.5/phone_id_map.txt
