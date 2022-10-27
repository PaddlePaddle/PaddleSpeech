#!/bin/bash

# To be run from one directory above this script.
. ./path.sh

nj=40
text=data/local/lm/text
lexicon=data/local/dict/lexicon.txt

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

# Check SRILM tools
if ! which ngram-count > /dev/null; then
    echo "srilm tools are not found, please download it and install it from: "
    echo "http://www.speech.sri.com/projects/srilm/download.html"
    echo "Then add the tools to your PATH"
    exit 1
fi

# This script takes no arguments.  It assumes you have already run
# aishell_data_prep.sh.
# It takes as input the files
# data/local/lm/text
# data/local/dict/lexicon.txt
dir=data/local/lm
mkdir -p $dir

cleantext=$dir/text.no_oov

# oov to <SPOKEN_NOISE>
# lexicon line: word char0 ... charn
# text line: utt word0 ... wordn -> line: <SPOKEN_NOISE> word0 ... wordn
text_dir=$(dirname $text)
split_name=$(basename $text)
./local/split_data.sh $text_dir $text $split_name $nj

utils/run.pl JOB=1:$nj $text_dir/split${nj}/JOB/${split_name}.no_oov.log \
  cat ${text_dir}/split${nj}/JOB/${split_name} \| awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } }
    {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<SPOKEN_NOISE> ");} } printf("\n");}' \
    \> ${text_dir}/split${nj}/JOB/${split_name}.no_oov || exit 1;
cat ${text_dir}/split${nj}/*/${split_name}.no_oov  > $cleantext

# compute word counts, sort in descending order
# line: count word
cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | sort --parallel=`nproc` | uniq -c | \
   sort --parallel=`nproc` -nr > $dir/word.counts || exit 1;

# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon (but not silence, we don't want it
# in the LM-- we'll add it optionally later).
cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
   sort --parallel=`nproc` | uniq -c | sort --parallel=`nproc` -nr > $dir/unigram.counts || exit 1;

# word with <s> </s>
cat $dir/unigram.counts | awk '{print $2}' | cat - <(echo "<s>"; echo "</s>" ) > $dir/wordlist

# hold out to compute ppl
heldout_sent=10000 # Don't change this if you want result to be comparable with kaldi_lm results

mkdir -p $dir
cat $cleantext | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
  head -$heldout_sent > $dir/heldout
cat $cleantext | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
  tail -n +$heldout_sent > $dir/train

ngram-count -text $dir/train -order 3 -limit-vocab -vocab $dir/wordlist -unk \
  -map-unk "<UNK>" -kndiscount -interpolate -lm $dir/lm.arpa
ngram -lm $dir/lm.arpa -ppl $dir/heldout