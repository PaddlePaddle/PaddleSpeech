#!/bin/bash
# Copyright 2015       Yajie Miao    (Carnegie Mellon University)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script compiles the lexicon and CTC tokens into FSTs. FST compiling slightly differs between the
# phoneme and character-based lexicons.
set -eo pipefail
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "usage: utils/fst/compile_lexicon_token_fst.sh <dict-src-dir> <tmp-dir> <lang-dir>"
  echo "e.g.: utils/fst/compile_lexicon_token_fst.sh data/local/dict data/local/lang_tmp data/lang"
  echo "<dict-src-dir> should contain the following files:"
  echo "lexicon.txt lexicon_numbers.txt units.txt"
  echo "options: "
  exit 1;
fi

srcdir=$1
tmpdir=$2
dir=$3
mkdir -p $dir $tmpdir

[ -f path.sh ] && . ./path.sh

cp $srcdir/units.txt $dir

# Add probabilities to lexicon entries. There is in fact no point of doing this here since all the entries have 1.0.
# But utils/make_lexicon_fst.pl requires a probabilistic version, so we just leave it as it is.
perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $srcdir/lexicon.txt > $tmpdir/lexiconp.txt || exit 1;

# Add disambiguation symbols to the lexicon. This is necessary for determinizing the composition of L.fst and G.fst.
# Without these symbols, determinization will fail.
# default first disambiguation is #1
ndisambig=`utils/fst/add_lex_disambig.pl $tmpdir/lexiconp.txt $tmpdir/lexiconp_disambig.txt`
# add #0 (#0 reserved for symbol in grammar).
ndisambig=$[$ndisambig+1];

( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) > $tmpdir/disambig.list

# Get the full list of CTC tokens used in FST. These tokens include <eps>, the blank <blk>,
# the actual model unit, and the disambiguation symbols.
cat $srcdir/units.txt | awk '{print $1}' > $tmpdir/units.list
(echo '<eps>';) | cat - $tmpdir/units.list $tmpdir/disambig.list | awk '{print $1 " " (NR-1)}' > $dir/tokens.txt

# ctc_token_fst_corrected is too big and too slow for character based chinese modeling,
# so here just use simple ctc_token_fst
utils/fst/ctc_token_fst.py --token_file $dir/tokens.txt | \
  fstcompile --isymbols=$dir/tokens.txt --osymbols=$dir/tokens.txt --keep_isymbols=false --keep_osymbols=false | \
  fstarcsort --sort_type=olabel > $dir/T.fst || exit 1;

# Encode the words with indices. Will be used in lexicon and language model FST compiling.
cat $tmpdir/lexiconp.txt | awk '{print $1}' | sort | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
    printf("<s> %d\n", NR+2);
    printf("</s> %d\n", NR+3);
  }' > $dir/words.txt || exit 1;

# Now compile the lexicon FST. Depending on the size of your lexicon, it may take some time.
token_disambig_symbol=`grep \#0 $dir/tokens.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 $dir/words.txt | awk '{print $2}'`

utils/fst/make_lexicon_fst.pl --pron-probs $tmpdir/lexiconp_disambig.txt 0 "sil" '#'$ndisambig | \
  fstcompile --isymbols=$dir/tokens.txt --osymbols=$dir/words.txt \
  --keep_isymbols=false --keep_osymbols=false |   \
  fstaddselfloops  "echo $token_disambig_symbol |" "echo $word_disambig_symbol |" | \
  fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;

echo "Lexicon and Token FSTs compiling succeeded"