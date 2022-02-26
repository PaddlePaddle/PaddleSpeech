#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


# The trials file is downloaded by local/make_voxceleb1.pl.
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=/home/dataset/sv/voxCeleb1_v2
nnet_dir=exp/xvector_nnet_1a
nj=28 # cpu cores 
stage=0

if [ $stage -le 0 ]; then
  echo "======================================================================================================"
  echo "=========================== Stage 0: Prepare the VoxCeleb1 dataset and trial=========================="
  echo "======================================================================================================"
  # This script reates data/voxceleb1_test and data/voxceleb1_train.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1_v2.pl $voxceleb1_root dev data/voxceleb1_train
  local/make_voxceleb1_v2.pl $voxceleb1_root test data/voxceleb1_test
  # This should give 1251 speakers and 153516 utterances.
  # utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train
  # utils/combine_data.sh data/train data/voxceleb1_train
  echo ""
fi

if [ $stage -le 1 ]; then
  echo "======================================================================================================"
  echo "=========================== Stage 1: Extract the VoxCeleb1 dataset feature ==========================="
  echo "======================================================================================================"
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in voxceleb1_train voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj ${nj} --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
  echo ""
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 2 ]; then
  echo "======================================================================================================"
  echo "=========================== Stage 2: Prepare the feature to train ===================================="
  echo "======================================================================================================"
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj ${nj} --cmd "$train_cmd" \
    data/voxceleb1_train data/voxceleb1_train_no_sil exp/voxceleb1_train_no_sil
  utils/fix_data_dir.sh data/voxceleb1_train_no_sil
  echo ""
fi

if [ $stage -le 3 ]; then
  echo "======================================================================================================"
  echo "=========================== Stage 3: Remove too short and throw spkeakers with fewer utterances ======"
  echo "======================================================================================================"
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv data/voxceleb1_train_no_sil/utt2num_frames data/voxceleb1_train_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/voxceleb1_train_no_sil/utt2num_frames.bak > data/voxceleb1_train_no_sil/utt2num_frames
  utils/filter_scp.pl data/voxceleb1_train_no_sil/utt2num_frames data/voxceleb1_train_no_sil/utt2spk > data/voxceleb1_train_no_sil/utt2spk.new
  mv data/voxceleb1_train_no_sil/utt2spk.new data/voxceleb1_train_no_sil/utt2spk
  utils/fix_data_dir.sh data/voxceleb1_train_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/voxceleb1_train_no_sil/spk2utt > data/voxceleb1_train_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/voxceleb1_train_no_sil/spk2num | utils/filter_scp.pl - data/voxceleb1_train_no_sil/spk2utt > data/voxceleb1_train_no_sil/spk2utt.new
  mv data/voxceleb1_train_no_sil/spk2utt.new data/voxceleb1_train_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/voxceleb1_train_no_sil/spk2utt > data/voxceleb1_train_no_sil/utt2spk

  utils/filter_scp.pl data/voxceleb1_train_no_sil/utt2spk data/voxceleb1_train_no_sil/utt2num_frames > data/voxceleb1_train_no_sil/utt2num_frames.new
  mv data/voxceleb1_train_no_sil/utt2num_frames.new data/voxceleb1_train_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  # This should give 1211 speakers and 138493 utterances to train.
  utils/fix_data_dir.sh data/voxceleb1_train_no_sil
fi

# Stages 6 through 8 are handled in run_xvector.sh
if [ $stage -le 8 ]; then
  echo "======================================================================================================"
  echo "=========================== Stage 4: Train the x-vector model ========================================"
  echo "======================================================================================================"
  local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
    --data data/voxceleb1_train_no_sil --nnet-dir $nnet_dir \
    --egs-dir $nnet_dir/egs
  echo ""
fi 

if [ $stage -le 9 ]; then
  echo "======================================================================================================"
  echo "=========================== Stage 9: Extract the train x-vector ======================================"
  echo "======================================================================================================"
  # Extract x-vectors for centering, LDA, and PLDA training.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 28 \
    $nnet_dir data/voxceleb1_train \
    $nnet_dir/voxceleb1_xvectors_train
  echo ""
fi 

if [ $stage -le 10 ]; then
  # Extract x-vectors used in the evaluation.
  echo "======================================================================================================"
  echo "=========================== Stage 10: Extract the test x-vector ======================================"
  echo "======================================================================================================"
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 28 \
    $nnet_dir data/voxceleb1_test \
    $nnet_dir/xvectors_voxceleb1_test
  echo ""
fi

if [ $stage -le 11 ]; then
  echo "======================================================================================================"
  echo "=========================== Stage 11: Train the plda ================================================="
  echo "======================================================================================================"
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnet_dir/voxceleb1_xvectors_train/log/compute_mean.log \
    ivector-mean scp:$nnet_dir/voxceleb1_xvectors_train/xvector.scp \
    $nnet_dir/voxceleb1_xvectors_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $nnet_dir/voxceleb1_xvectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/voxceleb1_xvectors_train/xvector.scp ark:- |" \
    ark:data/voxceleb1_train/utt2spk $nnet_dir/voxceleb1_xvectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/voxceleb1_xvectors_train/log/plda.log \
    ivector-compute-plda ark:data/voxceleb1_train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/voxceleb1_xvectors_train/xvector.scp ark:- | transform-vec $nnet_dir/voxceleb1_xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnet_dir/voxceleb1_xvectors_train/plda || exit 1;
  echo ""
fi

if [ $stage -le 12 ]; then
  echo "======================================================================================================"
  echo "=========================== Stage 12: Score the trial with plda ======================================"
  echo "======================================================================================================"
  $train_cmd exp/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/voxceleb1_xvectors_train/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/voxceleb1_xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/voxceleb1_xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/voxceleb1_xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/voxceleb1_xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test || exit 1;
  echo ""
fi

if [ $stage -le 13 ]; then
  echo "======================================================================================================"
  echo "=========================== Stage 13: Compute the EER  ==============================================="
  echo "======================================================================================================"
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
  echo "EER: ${eer}%"
  # EER: 7.508%
  echo ""
fi
