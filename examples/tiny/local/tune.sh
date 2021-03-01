#! /usr/bin/env bash

if [ $# != 1 ];then
    echo "usage: tune ckpt_path"
    exit 1
fi

# grid-search for hyper-parameters in language model
python3 -u ${BIN_DIR}/tune.py \
--device 'gpu' \
--nproc 1 \
--config conf/deepspeech2.yaml \
--num_batches=-1 \
--batch_size=128 \
--beam_size=500 \
--num_proc_bsearch=12 \
--num_alphas=45 \
--num_betas=8 \
--alpha_from=1.0 \
--alpha_to=3.2 \
--beta_from=0.1 \
--beta_to=0.45 \
--cutoff_prob=1.0 \
--cutoff_top_n=40 \
--checkpoint_path ${1}

if [ $? -ne 0 ]; then
    echo "Failed in tuning!"
    exit 1
fi


exit 0
