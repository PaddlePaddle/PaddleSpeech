#!/bin/bash

set -e
source path.sh

gpus=0,1
stage=0
stop_stage=100

conf_path=conf/default.yaml
train_output_path=exp/default
ckpt_name=snapshot_iter_331.pdz
source_path=test_wav/goat_01.wav
output_dir=vc_output

# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

# not ready now
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    ./local/preprocess.sh ${conf_path} || exit -1
fi
# not ready now
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `train_output_path/checkpoints/` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # synthesize, vocoder is pwgan by default
    CUDA_VISIBLE_DEVICES=${gpus} ./local/voice_conversion.sh ${conf_path} ${source_path} ${output_dir}|| exit -1
fi

