#! /usr/bin/env bash

BATCH_SIZE_PER_GPU=64
MIN_DURATION=6.0
MAX_DURATION=7.0

function join_by { local IFS="$1"; shift; echo "$*"; }

for NUM_GPUS in 16 8 4 2 1
do
  DEVICES=$(join_by , $(seq 0 $(($NUM_GPUS-1))))
  BATCH_SIZE=$(($BATCH_SIZE_PER_GPU))

  CUDA_VISIBLE_DEVICES=$DEVICES \
  python train.py \
  --batch_size=$BATCH_SIZE \
  --num_epoch=1 \
  --test_off=True \
  --min_duration=$MIN_DURATION \
  --max_duration=$MAX_DURATION > tmp.log 2>&1

  if [ $? -ne 0 ];then
      exit 1
  fi

  cat tmp.log  | grep "Time" | awk '{print "GPU Num: " "'"$NUM_GPUS"'" "	Time: "$2}'

  rm tmp.log
done
