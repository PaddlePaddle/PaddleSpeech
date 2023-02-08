#!/bin/bash
set -eo pipefail

#. path.sh

# attention, please replace the vocab is only for this script. 
# different acustic model has different vocab
ckpt_dir=data/model/asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model
unit=$ckpt_dir/vocab.txt       # vocab file, line: char/spm_pice
model_dir=$ckpt_dir/exp/deepspeech2_online/checkpoints/

stage=2
stop_stage=100
corpus=aishell
lexicon=data/lexicon.txt  # line: word ph0 ... phn, aishell/resource_aishell/lexicon.txt
text=data/text            # line: utt text, aishell/data_aishell/transcript/aishell_transcript_v0.8.txt

. utils/parse_options.sh

data=$PWD/data
mkdir -p $data

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
    if [ ! -f $data/speech.ngram.zh.tar.gz ];then
        # download ngram
        pushd $data
        wget -c http://paddlespeech.bj.bcebos.com/speechx/examples/ngram/zh/speech.ngram.zh.tar.gz
        tar xvzf speech.ngram.zh.tar.gz
        popd
    fi
fi

if [ ! -f $unit ]; then
    echo "$0: No such file $unit"
    exit 1;
fi

if ! which ngram-count; then
    # need srilm install
    pushd $MAIN_ROOT/tools
    make srilm.done
    popd
fi

echo "done."
mkdir -p data/local/dict
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Prepare dict
    # line: char/spm_pices
    cp $unit data/local/dict/units.txt

    if [ ! -f $lexicon ];then
       utils/text_to_lexicon.py --has_key true --text $text --lexicon $lexicon
        echo "Generate $lexicon from $text"
    fi

    # filter by vocab
    # line: word ph0 ... phn -> line: word char0 ... charn
    utils/fst/prepare_dict.py \
        --unit_file $unit \
        --in_lexicon ${lexicon} \
        --out_lexicon data/local/dict/lexicon.txt
fi

lm=data/local/lm
mkdir -p $lm

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Train ngram lm
    cp $text $lm/text
    local/aishell_train_lms.sh
    echo "build LM done."
fi

# build TLG
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # build T & L
  utils/fst/compile_lexicon_token_fst.sh \
      data/local/dict data/local/tmp data/local/lang
 
  # build G & TLG
  utils/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;

fi
