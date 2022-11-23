#! /usr/bin/env bash

stage=-1
stop_stage=100

source ${MAIN_ROOT}/utils/parse_options.sh

mkdir -p data
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}
LEXICON_NAME=$1

# download data, generate manifests
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    python3 ${TARGET_DIR}/thchs30/thchs30.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${TARGET_DIR}/thchs30"

    if [ $? -ne 0 ]; then
        echo "Prepare THCHS-30 failed. Terminated."
        exit 1
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # dump manifest to data/
    python3 ${MAIN_ROOT}/utils/dump_manifest.py --manifest-path=data/manifest.train --output-dir=data
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # copy files to data/dict to gen word.lexicon
    cp  ${TARGET_DIR}/thchs30/data_thchs30/lm_word/lexicon.txt data/dict/lm_word_lexicon_1
    cp  ${TARGET_DIR}/thchs30/resource/dict/lexicon.txt data/dict/lm_word_lexicon_2
    # copy phone.lexicon to data/dict
    cp  ${TARGET_DIR}/thchs30/data_thchs30/lm_phone/lexicon.txt data/dict/phone.lexicon
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # gen word.lexicon
    python local/gen_word2phone.py  --lexicon-files="data/dict/lm_word_lexicon_1 data/dict/lm_word_lexicon_2" --output-path=data/dict/word.lexicon
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # reorganize dataset for MFA
    if [ ! -d $EXP_DIR/thchs30_corpus ]; then
        echo "reorganizing thchs30 corpus..."
        python local/reorganize_thchs30.py --root-dir=data --output-dir=data/thchs30_corpus --script-type=$LEXICON_NAME
        echo "reorganization done."
    fi
fi

echo "THCHS-30  data preparation done."
exit 0
