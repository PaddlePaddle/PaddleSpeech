```
fstaddselfloops 'echo 4234 |' 'echo 123660 |' 
Lexicon and Token FSTs compiling succeeded
arpa2fst --read-symbol-table=data/lang_test/words.txt --keep-symbols=true - 
LOG (arpa2fst[5.5.0~1-5a37]:Read():arpa-file-parser.cc:94) Reading \data\ section.
LOG (arpa2fst[5.5.0~1-5a37]:Read():arpa-file-parser.cc:149) Reading \1-grams: section.
LOG (arpa2fst[5.5.0~1-5a37]:Read():arpa-file-parser.cc:149) Reading \2-grams: section.
LOG (arpa2fst[5.5.0~1-5a37]:Read():arpa-file-parser.cc:149) Reading \3-grams: section.
Checking how stochastic G is (the first of these numbers should be small):
fstisstochastic data/lang_test/G.fst 
0 -1.14386
fsttablecompose data/lang_test/L.fst data/lang_test/G.fst 
fstminimizeencoded 
fstdeterminizestar --use-log=true 
fsttablecompose data/lang_test/T.fst data/lang_test/LG.fst 
Composing decoding graph TLG.fst succeeded
Aishell build TLG done.
```