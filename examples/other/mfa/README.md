# Use Montreal-Forced-Aligner
Here is an example to use [MFA1.x](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner).
Run the following script to get started, for more detail, please see `run.sh`.
```bash
./run.sh
```
# Rhythm tags for MFA
If you want to get rhythm tags with duration through MFA tool, you may add flag `--rhy-with-duration` in the first two commands in `run.sh`
Note that only CSMSC dataset is supported so far, and we replace `#` with `sp` in rhythm tags for MFA.

# MFA for Cantonese language
First, go download these datasets [Guangzhou_Cantonese_Scripted_Speech_Corpus_Daily_Use_Sentence](https://paddlespeech.bj.bcebos.com/datasets/Cantonese/Guangzhou_Cantonese_Scripted_Speech_Corpus_Daily_Use_Sentence.zip) and [Guangzhou_Cantonese_Scripted_Speech_Corpus_in_Vehicle](https://paddlespeech.bj.bcebos.com/datasets/Cantonese/Guangzhou_Cantonese_Scripted_Speech_Corpus_in_Vehicle.zip) under `~/datasets/`.
Then,
```bash
./run_canton.sh
```
