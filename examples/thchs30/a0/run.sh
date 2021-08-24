#!/bin/bash
set -e
source path.sh
stage=0
stop_stage=100
EXP_DIR=exp
# LEXICON_NAME in {'phone', 'syllable', 'word'}
LEXICON_NAME='phone'
# set MFA num_jobs as half of machine's cpu core number
NUM_JOBS=$((`nproc`/2))
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

# download dataset、unzip and generate manifest 
# gen lexicon relink gen dump
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Start prepare thchs30 data for MFA ..."
    bash ./local/data.sh $LEXICON_NAME || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # run MFA
    if [ ! -d "$EXP_DIR/thchs30_alignment" ]; then
        echo "Start MFA training ..."
        mfa_train_and_align data/thchs30_corpus data/dict/$LEXICON_NAME.lexicon $EXP_DIR/thchs30_alignment -o $EXP_DIR/thchs30_model --clean --verbose --temp_directory exp/.mfa_train_and_align --num_jobs $NUM_JOBS
        echo "MFA training done! \nresults: $EXP_DIR/thchs30_alignment \nmodel: $EXP_DIR/thchs30_model\n"
    fi
fi







