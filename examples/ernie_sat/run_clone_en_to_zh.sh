#!/bin/bash

set -e
source path.sh

# en --> zh  的 语音合成
# 根据 Prompt_003_new 作为提示语音: This was not the show for me. 来合成:  '今天天气很好'
# 注: 输入的 new_str 需为中文汉字, 否则会通过预处理只保留中文汉字, 即合成预处理后的中文语音。

python local/inference.py \
    --task_name=cross-lingual_clone \
    --model_name=paddle_checkpoint_dual_mask_enzh \
    --uid=Prompt_003_new \
    --new_str='今天天气很好.' \
    --prefix='./prompt/dev/' \
    --source_lang=english \
    --target_lang=chinese \
    --output_name=pred_clone.wav \
    --voc=pwgan_aishell3 \
    --voc_config=download/pwg_aishell3_ckpt_0.5/default.yaml \
    --voc_ckpt=download/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
    --voc_stat=download/pwg_aishell3_ckpt_0.5/feats_stats.npy \
    --am=fastspeech2_csmsc \
    --am_config=download/fastspeech2_conformer_baker_ckpt_0.5/conformer.yaml \
    --am_ckpt=download/fastspeech2_conformer_baker_ckpt_0.5/snapshot_iter_76000.pdz \
    --am_stat=download/fastspeech2_conformer_baker_ckpt_0.5/speech_stats.npy \
    --phones_dict=download/fastspeech2_conformer_baker_ckpt_0.5/phone_id_map.txt
