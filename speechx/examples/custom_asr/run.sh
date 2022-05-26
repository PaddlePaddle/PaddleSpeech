#!/bin/bash
set +x
set -e

export GLOG_logtostderr=1

. ./path.sh || exit 1;

# ds2 means deepspeech2 (acoutic model type)
dir=$PWD/exp/ds2_graph_with_slot
data=$PWD/data
stage=0
stop_stage=10

mkdir -p $dir

model_dir=$PWD/resource/model
vocab=$model_dir/vocab.txt
cmvn=$data/cmvn.ark
text_with_slot=$data/text_with_slot
resource=$PWD/resource
# download resource
if [ ! -f $cmvn ]; then
    wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/resource.tar.gz
    tar xzfv resource.tar.gz
    ln -s ./resource/data .
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # make dict
  unit_file=$vocab
  mkdir -p $dir/local/dict
  cp $unit_file $dir/local/dict/units.txt
  cp $text_with_slot $dir/train_text
  utils/fst/prepare_dict.py --unit_file $unit_file --in_lexicon $data/lexicon.txt \
    --out_lexicon $dir/local/dict/lexicon.txt
  # add slot to lexicon, just in case the lm training script filter the slot.
  echo "<MONEY_SLOT> 一" >> $dir/local/dict/lexicon.txt
  echo "<DATE_SLOT> 一" >> $dir/local/dict/lexicon.txt
  echo "<ADDRESS_SLOT> 一" >> $dir/local/dict/lexicon.txt
  echo "<YEAR_SLOT> 一" >> $dir/local/dict/lexicon.txt
  echo "<TIME_SLOT> 一" >> $dir/local/dict/lexicon.txt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # train lm
  lm=$dir/local/lm
  mkdir -p $lm
  # this script is different with the common lm training script
  local/train_lm_with_slot.sh
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # make T & L
  local/compile_lexicon_token_fst.sh $dir/local/dict $dir/local/tmp $dir/local/lang
  mkdir -p $dir/local/lang_test
  # make slot graph
  local/mk_slot_graph.sh $resource/graph $dir/local/lang_test
  # make TLG
  local/mk_tlg_with_slot.sh $dir/local/lm $dir/local/lang $dir/local/lang_test || exit 1;
  mv $dir/local/lang_test/TLG.fst $dir/local/lang/
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # test TLG
  model_dir=$PWD/resource/model
  cmvn=$data/cmvn.ark
  wav_scp=$data/wav.scp
  graph=$dir/local/lang

  recognizer_test_main \
    --wav_rspecifier=scp:$wav_scp \
    --cmvn_file=$cmvn \
    --streaming_chunk=30 \
    --use_fbank=true \
    --model_path=$model_dir/avg_10.jit.pdmodel \
    --param_path=$model_dir/avg_10.jit.pdiparams \
    --model_cache_shapes="5-1-2048,5-1-2048" \
    --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
    --word_symbol_table=$graph/words.txt \
    --graph_path=$graph/TLG.fst --max_active=7500 \
    --acoustic_scale=12 \
    --result_wspecifier=ark,t:./exp/result_run.txt

    # the data/wav.trans is the label.
    utils/compute-wer.py --char=1 --v=1 data/wav.trans exp/result_run.txt > exp/wer_run
    tail -n 7 exp/wer_run
fi
