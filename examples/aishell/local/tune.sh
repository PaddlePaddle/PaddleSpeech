#! /usr/bin/env bash

# grid-search for hyper-parameters in language model
python3 -u ${BIN_DIR}/tune.py \
--device 'gpu' \
--nproc 1 \
--config conf/deepspeech2.yaml \
--num_batches=10 \
--batch_size=128 \
--beam_size=300 \
--num_proc_bsearch=8 \
--num_alphas=10 \
--num_betas=10 \
--alpha_from=0.0 \
--alpha_to=5.0 \
--beta_from=-6 \
--beta_to=6 \
--cutoff_prob=1.0 \
--cutoff_top_n=40 \
--checkpoint_path ${1}

if [ $? -ne 0 ]; then
    echo "Failed in tuning!"
    exit 1
fi


exit 0
