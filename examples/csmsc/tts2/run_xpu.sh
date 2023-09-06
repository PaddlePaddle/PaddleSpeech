#!/bin/bash

set -e
source path.sh

xpus=0,1
stage=0
stop_stage=100

conf_path=conf/default.yaml
train_output_path=exp/default
ckpt_name=snapshot_iter_76.pdz

# with the following command, you can choose the stage range you want to run
# such as `./run_xpu.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    ./local/preprocess.sh ${conf_path} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `train_output_path/checkpoints/` dir
    FLAGS_selected_xpus=${xpus} ./local/train_xpu.sh ${conf_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # synthesize, vocoder is pwgan by default
    FLAGS_selected_xpus=${xpus} ./local/synthesize_xpu.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # synthesize_e2e, vocoder is pwgan by default
    FLAGS_selected_xpus=${xpus} ./local/synthesize_e2e_xpu.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # inference with static model
    FLAGS_selected_xpus=${xpus} ./local/inference_xpu.sh ${train_output_path} || exit -1
fi
