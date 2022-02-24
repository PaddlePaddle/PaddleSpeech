#!/bin/bash
set -e

stage=0
voxceleb1_root=/mnt/dataset_12/sv/voxCeleb1_v2/

if [ $stage -le 0 ]; then
    echo "======================================================================================================"
    echo "=========================== Stage 0: Prepare the VoxCeleb1 dataset ==================================="
    echo "======================================================================================================"
    # prepare the data elapsed about 20s
    # the script will create the data/{dev,test}
    local/data.sh ${voxceleb1_root}|| exit 1;
fi
