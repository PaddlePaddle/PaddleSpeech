#!/bin/bash

. ./path.sh || exit 1;

. tools/parse_options.sh || exit 1;

data=/mnt/dataset/aishell

# Optionally, you can add LM and test it with runtime.
dir=./ds2_graph
dict=$dir/vocab.txt
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  # 7.1 Prepare dict
  unit_file=$dict
  mkdir -p $dir/local/dict
  cp $unit_file $dir/local/dict/units.txt
  tools/fst/prepare_dict.py $unit_file ${data}/resource_aishell/lexicon.txt \
    $dir/local/dict/lexicon.txt
  # Train lm
  lm=$dir/local/lm
  mkdir -p $lm
  tools/filter_scp.pl data/train/text \
    $data/data_aishell/transcript/aishell_transcript_v0.8.txt > $lm/text
  local/ds2_aishell_train_lms.sh
  # Build decoding TLG
  tools/fst/compile_lexicon_token_fst.sh \
    $dir/local/dict $dir/local/tmp $dir/local/lang
  tools/fst/make_tlg.sh $dir/local/lm $dir/local/lang $dir/lang_test || exit 1;
fi


