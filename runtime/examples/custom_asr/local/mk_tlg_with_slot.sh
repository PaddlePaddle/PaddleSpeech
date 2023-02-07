#!/bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

lm=$1
lang=$2
tgt_lang=$3

unset GREP_OPTIONS

sym=$lang/words.txt
arpa_lm=$lm/lm.arpa
# Compose the language model to FST
cat $arpa_lm | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   grep -v -i '<unk>' | \
   grep -v -i '<spoken_noise>' | \
   arpa2fst --read-symbol-table=$sym --keep-symbols=true - | fstprint | \
   utils/fst/eps2disambig.pl | utils/fst/s2eps.pl | fstcompile --isymbols=$sym \
     --osymbols=$sym --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > $tgt_lang/G_with_slot.fst

root_label=`grep ROOT $sym | awk '{print $2}'`
address_slot_label=`grep \<ADDRESS_SLOT\> $sym | awk '{print $2}'`
time_slot_label=`grep \<TIME_SLOT\> $sym | awk '{print $2}'`
date_slot_label=`grep \<DATE_SLOT\> $sym | awk '{print $2}'`
money_slot_label=`grep \<MONEY_SLOT\> $sym | awk '{print $2}'`
year_slot_label=`grep \<YEAR_SLOT\> $sym | awk '{print $2}'`

fstisstochastic $tgt_lang/G_with_slot.fst

fstreplace --epsilon_on_replace $tgt_lang/G_with_slot.fst \
  $root_label $tgt_lang/address_slot.fst $address_slot_label \
  $tgt_lang/date_slot.fst $date_slot_label \
  $tgt_lang/money_slot.fst $money_slot_label \
  $tgt_lang/time_slot.fst $time_slot_label \
  $tgt_lang/year_slot.fst $year_slot_label $tgt_lang/G.fst

fstisstochastic $tgt_lang/G.fst

# Compose the token, lexicon and language-model FST into the final decoding graph
fsttablecompose $lang/L.fst $tgt_lang/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstarcsort --sort_type=ilabel > $tgt_lang/LG.fst || exit 1;
fsttablecompose $lang/T.fst $tgt_lang/LG.fst > $tgt_lang/TLG.fst || exit 1;
rm $tgt_lang/LG.fst

echo "Composing decoding graph TLG.fst succeeded"