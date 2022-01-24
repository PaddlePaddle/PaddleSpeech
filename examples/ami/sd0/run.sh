#!/bin/bash

. path.sh || exit 1;
set -e

stage=1


. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

if [ ${stage} -le 1 ]; then
    # prepare data
    bash ./local/data.sh || exit -1
fi