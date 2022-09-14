#!/bin/bash

set -e
source path.sh

gpus=0,1,2,3
stage=0
stop_stage=100

conf_path=conf/default.yaml
train_output_path=exp/default
ckpt_name=snapshot_iter_153.pdz
add_blank=true
ref_audio_dir=ref_audio
src_audio_path=''

# not include ".pdparams" here
ge2e_ckpt_path=./ge2e_ckpt_0.3/step-3000000

# include ".pdparams" here
ge2e_params_path=${ge2e_ckpt_path}.pdparams

# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    CUDA_VISIBLE_DEVICES=${gpus} ./local/preprocess.sh ${conf_path} ${add_blank}  ${ge2e_ckpt_path} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `train_output_path/checkpoints/` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/voice_cloning.sh ${conf_path} ${train_output_path} ${ckpt_name} \
        ${ge2e_params_path} ${add_blank} ${ref_audio_dir} ${src_audio_path} || exit -1
fi
