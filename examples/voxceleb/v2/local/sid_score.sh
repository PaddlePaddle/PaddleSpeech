#! /usr/bin/env bash
enroll_vector=$1
test_vector=$2
trial=$3
stage=-1

# if [ ${stage} -le 0 ]; then
#     python3 ./local/sid_score.py \
#                 --enroll ${enroll_vector} \
#                 --test ${test_vector} \
#                 --trial ${trial}

# fi

if [ ${stage} -le 1 ]; then
    python3 ./local/compute_eer.py exp/ecapa_tdnn/model/scores
fi