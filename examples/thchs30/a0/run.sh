#!/bin/bash
set -e
source path.sh
stage=0
stop_stage=100
EXP_DIR=exp
# LEXICON_NAME in {'phone', 'syllable', 'text'}
LEXICON_NAME='phone'
# get machine's cpu core number
NUM_JOBS=`grep 'processor' /proc/cpuinfo | sort -u | wc -l`
NUM_JOBS=$((NUM_JOBS/2))
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

# download dataset、unzip and generate manifest 
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash ./local/data.sh || exit -1
fi

# reorganize dataset for MFA
if [ ! -d $EXP_DIR/thchs30_corpus ]; then
    echo "reorganizing thchs30 corpus..."
    python local/recorganize_thchs30.py --root-dir=./data --output-dir=$EXP_DIR/thchs30_corpus --script-type=$LEXICON_NAME
    echo "reorganization done."
fi
# MFA is in tools
export PATH="${MAIN_ROOT}/tools/montreal-forced-aligner/bin"

if [ ! -d "$EXP_DIR/thchs30_alignment" ]; then
    echo "Start MFA training..."
    mfa_train_and_align $EXP_DIR/thchs30_corpus "$EXP_DIR/$LEXICON_NAME.lexicon" $EXP_DIR/thchs30_alignment -o $EXP_DIR/thchs30_model --clean --verbose --temp_directory exp/.mfa_train_and_align --num_jobs $NUM_JOBS
    echo "training done! \nresults: $EXP_DIR/thchs30_alignment \nmodel: $EXP_DIR/thchs30_model\n"
fi







