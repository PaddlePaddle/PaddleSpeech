#!/usr/bin/env bash

data=$1
feat_scp=$2
split_feat_name=$3
numsplit=$4


if [[ ! $numsplit -gt 0 ]]; then
  echo "Invalid num-split argument";
  exit 1;
fi

directories=$(for n in `seq $numsplit`; do echo $data/split${numsplit}/$n; done)
feat_split_scp=$(for n in `seq $numsplit`; do echo $data/split${numsplit}/$n/${split_feat_name}; done)
echo $feat_split_scp
# if this mkdir fails due to argument-list being too long, iterate.
if ! mkdir -p $directories >&/dev/null; then
  for n in `seq $numsplit`; do
    mkdir -p $data/split${numsplit}/$n
  done
fi

utils/split_scp.pl $feat_scp $feat_split_scp
