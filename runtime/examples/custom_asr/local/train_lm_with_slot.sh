#!/bin/bash

# To be run from one directory above this script.
. ./path.sh
src=ds2_graph_with_slot
text=$src/train_text
lexicon=$src/local/dict/lexicon.txt

dir=$src/local/lm
mkdir -p $dir

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

# Check SRILM tools
if ! which ngram-count > /dev/null; then
  pushd $MAIN_ROOT/tools
  make srilm.done
  popd
fi

# This script takes no arguments.  It assumes you have already run
# It takes as input the files
# data/local/lm/text
# data/local/dict/lexicon.txt


cleantext=$dir/text.no_oov

cat $text | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } }
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<SPOKEN_NOISE> ");} } printf("\n");}' \
  > $cleantext || exit 1;

cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq -c | \
   sort -nr > $dir/word.counts || exit 1;
# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon (but not silence, we don't want it
# in the LM-- we'll add it optionally later).
cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
   sort | uniq -c | sort -nr > $dir/unigram.counts || exit 1;

# filter the words which are not in the text
cat $dir/unigram.counts | awk '$1>1{print $0}' | awk '{print $2}' | cat - <(echo "<s>"; echo "</s>" ) > $dir/wordlist

# kaldi_lm results
mkdir -p $dir
cat $cleantext | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' > $dir/train

ngram-count -text $dir/train -order 3 -limit-vocab -vocab $dir/wordlist -unk \
  -map-unk "<UNK>" -gt3max 0 -gt2max 0 -gt1max 0 -lm $dir/lm.arpa

#ngram-count -text $dir/train -order 3 -limit-vocab -vocab $dir/wordlist -unk \
#  -map-unk "<UNK>" -lm $dir/lm2.arpa