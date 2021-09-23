#!/usr/bin/env bash
set -o errexit

if [ $# != 2 ]; then
  echo "Usage: split_json.sh manifest num-to-split"
  exit 1
fi

data=data

jsonfile=$1
numsplit=$2

if [ $numsplit -le 0 ]; then
  echo "Invalid num-split argument $numsplit";
  exit 1;
fi

n=0;
jsons=""

# utilsscripts/get_split.pl returns "0 1 2 3" or "00 01 .. 18 19" or whatever.
# for n in `get_splits.pl $numsplit`; do
for n in `seq 1 $numsplit`; do  # Changed this to usual number sequence -Arnab
  mkdir -p $data/split$numsplit/$n
  jsons="$jsons $data/split$numsplit/$n/${jsonfile}"
done

split_scp.pl $data/${jsonfile} $jsons

exit 0
