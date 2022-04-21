#!/usr/bin/env bash

set -eo pipefail

data=$1
scp=$2
split_name=$3
numsplit=$4

# save in $data/split{n}
# $scp to split
# 

if [[ ! $numsplit -gt 0 ]]; then
  echo "Invalid num-split argument";
  exit 1;
fi

directories=$(for n in `seq $numsplit`; do echo $data/split${numsplit}/$n; done)
scp_splits=$(for n in `seq $numsplit`; do echo $data/split${numsplit}/$n/${split_name}; done)

# if this mkdir fails due to argument-list being too long, iterate.
if ! mkdir -p $directories >&/dev/null; then
  for n in `seq $numsplit`; do
    mkdir -p $data/split${numsplit}/$n
  done
fi

echo "utils/split_scp.pl $scp $scp_splits"
utils/split_scp.pl $scp $scp_splits
