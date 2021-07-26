#!/bin/bash
set -e
source path.sh

stage=0
stop_stage=100
conf_path=conf/conformer.yaml
avg_num=20

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

avg_ckpt=avg_${avg_num}
ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
echo "checkpoint name ${ckpt}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash ./local/data.sh || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./local/train.sh ${conf_path}  ${ckpt}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    avg.sh exp/${ckpt}/checkpoints ${avg_num}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # test ckpt avg_n
    CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # ctc alignment of test data
    CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi

# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#     # export ckpt avg_n
#     CUDA_VISIBLE_DEVICES=0 ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit
# fi

 # Optionally, you can add LM and test it with runtime.
 if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
     # 7.1 Prepare dict
     unit_file=data/vocab.txt
     mkdir -p data/local/dict
     cp $unit_file data/local/dict/units.txt
     utils/fst/prepare_dict.py $unit_file ${data}/resource_aishell/lexicon.txt \
         data/local/dict/lexicon.txt
        
     # 7.2 Train lm
     lm=data/local/lm
     mkdir -p $lm
     utils/filter_scp.pl data/train/text \
          $data/data_aishell/transcript/aishell_transcript_v0.8.txt > $lm/text
     local/aishell_train_lms.sh

     # 7.3 Build decoding TLG
     utils/fst/compile_lexicon_token_fst.sh \
         data/local/dict data/local/tmp data/local/lang
     utils/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;

    #  # 7.4 Decoding with runtime
    #  # reverse_weight only works for u2++ model and only left to right decoder is used when it is set to 0.0.
    #  dir=exp/conformer
    #  reverse_weight=0.0
    #  chunk_size=-1
    #  ./tools/decode.sh --nj 16 \
    #      --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    #      --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
    #      --reverse_weight $reverse_weight --chunk_size $chunk_size \
    #      --fst_path data/lang_test/TLG.fst \
    #      data/test/wav.scp data/test/text $dir/final.zip \
    #      data/lang_test/words.txt $dir/lm_with_runtime
    #  # See $dir/lm_with_runtime for wer
 fi
