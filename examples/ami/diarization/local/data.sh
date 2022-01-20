#!/bin/bash

stage=1

data_folder=/home/data/ami/amicorpus #e.g., /path/to/amicorpus/
manual_annot_folder=/home/data/ami/ami_public_manual_1.6.2 #e.g., /path/to/ami_public_manual_1.6.2/

save_folder=results
ref_rttm_dir=results/ref_rttms
meta_data_dir=results/metadata

set=L

. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;
set -u
set -o pipefail

mkdir -p ${save_folder}

if [ ${stage} -le 0 ]; then
    # Download AMI corpus, You need around 10GB of free space to get whole data
    # The signals are too large to package in this way,
    # so you need to use the chooser to indicate which ones you wish to download
    echo "Please follow https://groups.inf.ed.ac.uk/ami/download/ to download the data."
    echo "Annotations: AMI manual annotations v1.6.2 "
    echo "Signals: Scenario Meetings/Non Scenario Meetings, some sessions recommended but not all"
    echo "media streams: Headset mix, recommended first"
    exit 0;
fi

if [ ${stage} -le 1 ]; then
    echo "AMI Data preparation"

    python local/ami_prepare.py  --data_folder ${data_folder} \
            --manual_annot_folder ${manual_annot_folder} \
            --save_folder ${save_folder} --ref_rttm_dir ${ref_rttm_dir} \
            --meta_data_dir ${meta_data_dir} 
    
    if [ $? -ne 0 ]; then
        echo "Prepare AMI failed. Please check log message."
        exit 1
    fi
            
fi

echo "AMI data preparation done."
exit 0
