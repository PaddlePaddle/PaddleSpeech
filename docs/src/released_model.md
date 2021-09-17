# Released Models

## Acoustic Model Released
Acoustic Model | Training Data | Token-based | Size | Descriptions | CER  | Hours of speech
:-------------:| :------------:| :-----: | -----: | :----------------- | :---------- | :---------
[Ds2 Online Aishell Model](https://deepspeech.bj.bcebos.com/release2.1/aishell/s0/aishell.s0.ds_online.5rnn.debug.tar.gz) | Aishell Dataset | Char-based | 345 MB  | 2 Conv + 5 LSTM layers with only forward direction | 0.0824 | 151 h
[Ds2 Offline Aishell Model](https://deepspeech.bj.bcebos.com/release2.1/aishell/s0/aishell.s0.ds2.offline.cer6p65.release.tar.gz)| Aishell Dataset | Char-based | 306 MB | 2 Conv + 3 bidirectional gru layers| 0.065 | 151 h


## Language Model Released

Language Model | Training Data | Token-based | Size | Descriptions
:-------------:| :------------:| :-----: | -----: | :-----------------
[English LM](https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm) |  [CommonCrawl(en.00)](http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.00.deduped.xz) | Word-based | 8.3 GB | Pruned with 0 1 1 1 1; <br/> About 1.85 billion n-grams; <br/> 'trie'  binary with '-a 22 -q 8 -b 8'
[Mandarin LM Small](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm) | Baidu Internal Corpus | Char-based | 2.8 GB | Pruned with 0 1 2 4 4; <br/> About 0.13 billion n-grams; <br/> 'probing' binary with default settings
[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) | Baidu Internal Corpus | Char-based | 70.4 GB | No Pruning; <br/> About 3.7 billion n-grams; <br/> 'probing' binary with default settings
