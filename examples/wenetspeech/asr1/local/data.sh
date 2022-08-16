#!/bin/bash

# Copyright 2021  Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 NPU, ASLP Group (Author: Qijie Shao)
#
# Modified from wenet(https://github.com/wenet-e2e/wenet)

stage=-1
stop_stage=100

# Use your own data path. You need to download the WenetSpeech dataset by yourself.
wenetspeech_data_dir=./wenetspeech
# Make sure you have 1.2T for ${shards_dir}
shards_dir=./wenetspeech_shards

#wenetspeech training set
set=L
train_set=train_`echo $set | tr 'A-Z' 'a-z'`
dev_set=dev
test_sets="test_net test_meeting"

cmvn=true
cmvn_sampling_divisor=20 # 20 means 5% of the training data to estimate cmvn


. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;
set -u
set -o pipefail


mkdir -p data
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data
    echo "Please follow https://github.com/wenet-e2e/WenetSpeech to download the data."
    exit 0;
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Data preparation"
    local/wenetspeech_data_prep.sh \
        --train-subset $set \
        $wenetspeech_data_dir \
        data || exit 1;
fi

dict=data/lang_char/vocab.txt
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Make a dictionary"
    echo "dictionary: ${dict}"
    mkdir -p $(dirname $dict)
    echo "<blank>" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk>" >> ${dict} # <unk> must be 1
    echo "▁" >> ${dict} # ▁ is for space
    utils/text2token.py -s 1 -n 1 --space "▁" data/${train_set}/text \
        | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' \
        | grep -v "▁" \
        | awk '{print $0}' >> ${dict} \
        || exit 1;
    echo "<eos>" >> $dict
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Compute cmvn"
  # Here we use all the training data, you can sample some some data to save time
  # BUG!!! We should use the segmented data for CMVN
  if $cmvn; then
    full_size=`cat data/${train_set}/wav.scp | wc -l`
    sampling_size=$((full_size / cmvn_sampling_divisor))
    shuf -n $sampling_size data/$train_set/wav.scp \
      > data/$train_set/wav.scp.sampled
    python3 utils/compute_cmvn_stats.py \
    --num_workers 16 \
    --train_config $train_config \
    --in_scp data/$train_set/wav.scp.sampled \
    --out_cmvn data/$train_set/mean_std.json \
    || exit 1;
  fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Making shards, please wait..."
  RED='\033[0;31m'
  NOCOLOR='\033[0m'
  echo -e "It requires ${RED}1.2T ${NOCOLOR}space for $shards_dir, please make sure you have enough space"
  echo -e "It takes about ${RED}12 ${NOCOLOR}hours with 32 threads"
  for x in $dev_set $test_sets ${train_set}; do
    dst=$shards_dir/$x
    mkdir -p $dst
    utils/make_filted_shard_list.py --num_node 1 --num_gpus_per_node 8 --num_utts_per_shard 1000 \
      --do_filter --resample 16000  \
      --num_threads 32 --segments data/$x/segments \
      data/$x/wav.scp data/$x/text \
      $(realpath $dst) data/$x/data.list
  done
fi

echo "Wenetspeech data preparation done."
exit 0
