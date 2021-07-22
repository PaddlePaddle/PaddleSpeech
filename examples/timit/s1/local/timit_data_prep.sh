#!/usr/bin/env bash

# Copyright 2013   (Authors: Bagher BabaAli, Daniel Povey, Arnab Ghoshal)
#           2014   Brno University of Technology (Author: Karel Vesely)
# Apache 2.0.

if [ $# -ne 1 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi

dir=`pwd`/data/local
mkdir -p $dir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

function error_exit () {
  echo -e "$@" >&2; exit 1;
}
PROG=$(basename $0)

[ -f $conf/test_spk.list ] || error_exit "$PROG line $LINENO: Eval-set speaker list not found.";
[ -f $conf/dev_spk.list ] || error_exit "$PROG line $LINENO: dev-set speaker list not found.";

# First check if the train & test directories exist (these can either be upper-
# or lower-cased
if [ ! -d $*/TRAIN -o ! -d $*/TEST ] && [ ! -d $*/train -o ! -d $*/test ]; then
  echo "timit_data_prep.sh: Spot check of command line argument failed"
  echo "Command line argument must be absolute pathname to TIMIT directory"
  echo "with name like /export/corpora5/LDC/LDC93S1/timit/TIMIT"
  exit 1;
fi

# Now check what case the directory structure is
uppercased=false
train_dir=train
test_dir=test
if [ -d $*/TRAIN ]; then
  uppercased=true
  train_dir=TRAIN
  test_dir=TEST
fi

tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

# Get the list of speakers. The list of speakers in the 24-speaker core test
# set and the 50-speaker development set must be supplied to the script. All
# speakers in the 'train' directory are used for training.
if $uppercased; then
  tr '[:lower:]' '[:upper:]' < $conf/dev_spk.list > $tmpdir/dev_spk
  tr '[:lower:]' '[:upper:]' < $conf/test_spk.list > $tmpdir/test_spk
  ls -d "$*"/TRAIN/DR*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
else
  tr '[:upper:]' '[:lower:]' < $conf/dev_spk.list > $tmpdir/dev_spk
  tr '[:upper:]' '[:lower:]' < $conf/test_spk.list > $tmpdir/test_spk
  ls -d "$*"/train/dr*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
fi

cd $dir
for x in train dev test; do
  # First, find the list of audio files (use only si & sx utterances).
  # Note: train & test sets are under different directories, but doing find on
  # both and grepping for the speakers will work correctly.
  find $*/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.WAV' \
    | grep -f $tmpdir/${x}_spk > ${x}_sph.flist

  sed -e 's:.*/\(.*\)/\(.*\).\(WAV\|wav\)$:\1_\2:' ${x}_sph.flist \
    > $tmpdir/${x}_sph.uttids
  paste $tmpdir/${x}_sph.uttids ${x}_sph.flist \
    | sort -k1,1 > ${x}_sph.scp

  cat ${x}_sph.scp | awk '{print $1}' > ${x}.uttids

  # Now, Convert the transcripts into our format (no normalization yet)
  # Get the transcripts: each line of the output contains an utterance
  # ID followed by the transcript.
  find $*/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.PHN' \
    | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_phn.flist
  sed -e 's:.*/\(.*\)/\(.*\).\(PHN\|phn\)$:\1_\2:' $tmpdir/${x}_phn.flist \
    > $tmpdir/${x}_phn.uttids
  while read line; do
    [ -f $line ] || error_exit "Cannot find transcription file '$line'";
    cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;'
  done < $tmpdir/${x}_phn.flist > $tmpdir/${x}_phn.trans
  paste $tmpdir/${x}_phn.uttids $tmpdir/${x}_phn.trans \
    | sort -k1,1 > ${x}.trans

  # Do normalization steps.
  cat ${x}.trans | $local/timit_norm_trans.pl -i - -m $conf/phones.60-48-39.map -to 39 | sort > $x.text || exit 1;

done

echo "Data preparation succeeded"