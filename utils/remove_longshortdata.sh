#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

maxframes=2000
minframes=10
maxchars=200
minchars=0
nlsyms=""
no_feat=false
trans_type=char

help_message="usage: $0 olddatadir newdatadir"

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
    echo "${help_message}"
    exit 1;
fi

sdir=$1
odir=$2
mkdir -p ${odir}/tmp

if [ ${no_feat} = true ]; then
    # for machine translation
    cut -d' ' -f 1 ${sdir}/text > ${odir}/tmp/reclist1
else
    echo "extract utterances having less than $maxframes or more than $minframes frames"
    utils/data/get_utt2num_frames.sh ${sdir}
    < ${sdir}/utt2num_frames  awk -v maxframes="$maxframes" '{ if ($2 < maxframes) print }' \
        | awk -v minframes="$minframes" '{ if ($2 > minframes) print }' \
        | awk '{print $1}' > ${odir}/tmp/reclist1
fi

echo "extract utterances having less than $maxchars or more than $minchars characters"
# counting number of chars. Use (NF - 1) instead of NF to exclude the utterance ID column
if [ -z ${nlsyms} ]; then
text2token.py -s 1 -n 1 ${sdir}/text --trans_type ${trans_type} \
    | awk -v maxchars="$maxchars" '{ if (NF - 1 < maxchars) print }' \
    | awk -v minchars="$minchars" '{ if (NF - 1 > minchars) print }' \
    | awk '{print $1}' > ${odir}/tmp/reclist2
else
text2token.py -l ${nlsyms} -s 1 -n 1 ${sdir}/text --trans_type ${trans_type} \
    | awk -v maxchars="$maxchars" '{ if (NF - 1 < maxchars) print }' \
    | awk -v minchars="$minchars" '{ if (NF - 1 > minchars) print }' \
    | awk '{print $1}' > ${odir}/tmp/reclist2
fi

# extract common lines
comm -12 <(sort ${odir}/tmp/reclist1) <(sort ${odir}/tmp/reclist2) > ${odir}/tmp/reclist

reduce_data_dir.sh ${sdir} ${odir}/tmp/reclist ${odir}
utils/fix_data_dir.sh ${odir}

oldnum=$(wc -l ${sdir}/feats.scp | awk '{print $1}')
newnum=$(wc -l ${odir}/feats.scp | awk '{print $1}')
echo "change from $oldnum to $newnum"
