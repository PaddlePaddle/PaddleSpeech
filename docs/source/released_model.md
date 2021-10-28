# Released Models

## Speech-To-Text Models
### Acoustic Model Released in paddle 2.X
Acoustic Model | Training Data | Token-based | Size | Descriptions | CER | WER | Hours of speech
:-------------:| :------------:| :-----: | -----: | :----------------- |:--------- | :---------- | :---------
[Ds2 Online Aishell Model](https://deepspeech.bj.bcebos.com/release2.1/aishell/s0/aishell.s0.ds_online.5rnn.debug.tar.gz) | Aishell Dataset | Char-based | 345 MB  | 2 Conv + 5 LSTM layers with only forward direction | 0.0824 |-| 151 h
[Ds2 Offline Aishell Model](https://deepspeech.bj.bcebos.com/release2.1/aishell/s0/aishell.s0.ds2.offline.cer6p65.release.tar.gz)| Aishell Dataset | Char-based | 306 MB | 2 Conv + 3 bidirectional GRU layers| 0.065 |-| 151 h
[Conformer Online Aishell Model](https://deepspeech.bj.bcebos.com/release2.1/aishell/s1/aishell.chunk.release.tar.gz) | Aishell Dataset | Char-based | 283 MB  | Encoder:Conformer, Decoder:Transformer, Decoding method: Attention + CTC | 0.0594 |-| 151 h
[Conformer Offline Aishell Model](https://deepspeech.bj.bcebos.com/release2.1/aishell/s1/aishell.release.tar.gz) | Aishell Dataset | Char-based | 284 MB  | Encoder:Conformer, Decoder:Transformer, Decoding method: Attention | 0.0547 |-| 151 h
[Conformer Librispeech Model](https://deepspeech.bj.bcebos.com/release2.1/librispeech/s1/conformer.release.tar.gz) | Librispeech Dataset | Word-based | 287 MB  | Encoder:Conformer, Decoder:Transformer, Decoding method: Attention |-| 0.0325 | 960 h
[Transformer Librispeech Model](https://deepspeech.bj.bcebos.com/release2.1/librispeech/s1/transformer.release.tar.gz) | Librispeech Dataset | Word-based | 195 MB  | Encoder:Transformer, Decoder:Transformer, Decoding method: Attention |-| 0.0544 | 960 h

### Acoustic Model Transformed from paddle 1.8
Acoustic Model | Training Data | Token-based | Size | Descriptions | CER | WER | Hours of speech
:-------------:| :------------:| :-----: | -----: | :----------------- | :---------- | :---------- | :---------
[Ds2 Offline Aishell model](https://deepspeech.bj.bcebos.com/mandarin_models/aishell_model_v1.8_to_v2.x.tar.gz)|Aishell Dataset| Char-based| 234 MB| 2 Conv + 3 bidirectional GRU layers| 0.0804 |-| 151 h|
[Ds2 Offline Librispeech model](https://deepspeech.bj.bcebos.com/eng_models/librispeech_v1.8_to_v2.x.tar.gz)|Librispeech Dataset| Word-based| 307 MB| 2 Conv + 3 bidirectional sharing weight RNN layers |-| 0.0685| 960 h|
[Ds2 Offline Baidu en8k model](https://deepspeech.bj.bcebos.com/eng_models/baidu_en8k_v1.8_to_v2.x.tar.gz)|Baidu Internal English Dataset| Word-based| 273 MB| 2 Conv + 3 bidirectional GRU layers |-| 0.0541 | 8628 h|

### Language Model Released

Language Model | Training Data | Token-based | Size | Descriptions
:-------------:| :------------:| :-----: | -----: | :-----------------
[English LM](https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm) |  [CommonCrawl(en.00)](http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.00.deduped.xz) | Word-based | 8.3 GB | Pruned with 0 1 1 1 1; <br/> About 1.85 billion n-grams; <br/> 'trie'  binary with '-a 22 -q 8 -b 8'
[Mandarin LM Small](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm) | Baidu Internal Corpus | Char-based | 2.8 GB | Pruned with 0 1 2 4 4; <br/> About 0.13 billion n-grams; <br/> 'probing' binary with default settings
[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) | Baidu Internal Corpus | Char-based | 70.4 GB | No Pruning; <br/> About 3.7 billion n-grams; <br/> 'probing' binary with default settings

## Text-To-Speech Models
### Acoustic Models
Model Type | Dataset| Example Link | Pretrained Models
:-------------:| :------------:| :-----: | :-----
Tacotron2|LJSpeech|[tacotron2-vctk](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/ljspeech/tts0)|[tacotron2_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_ckpt_0.3.zip)
TransformerTTS| LJSpeech| [transformer-ljspeech](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/ljspeech/tts1)|[transformer_tts_ljspeech_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_ckpt_0.4.zip)
SpeedySpeech| CSMSC | [speedyspeech-csmsc](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/csmsc/tts2) |[speedyspeech_nosil_baker_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/speedyspeech_nosil_baker_ckpt_0.5.zip)
FastSpeech2| CSMSC |[fastspeech2-csmsc](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/csmsc/tts3)|[fastspeech2_nosil_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_baker_ckpt_0.4.zip)
FastSpeech2| AISHELL-3 |[fastspeech2-aishell3](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/aishell3/tts3)|[fastspeech2_nosil_aishell3_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_aishell3_ckpt_0.4.zip)
FastSpeech2| LJSpeech |[fastspeech2-ljspeech](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/ljspeech/tts3)|[fastspeech2_nosil_ljspeech_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_ljspeech_ckpt_0.5.zip)
FastSpeech2| VCTK |[fastspeech2-csmsc](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/vctk/tts3)|[fastspeech2_nosil_vctk_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_vctk_ckpt_0.5.zip)


### Vocoders

Model Type | Dataset| Example Link | Pretrained Models
:-------------:| :------------:| :-----: | :-----
WaveFlow| LJSpeech |[waveflow-ljspeech](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/ljspeech/voc0)|[waveflow_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_ljspeech_ckpt_0.3.zip)
Parallel WaveGAN| CSMSC |[PWGAN-csmsc](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/csmsc/voc1)|[pwg_baker_ckpt_0.4.zip.](https://paddlespeech.bj.bcebos.com/Parakeet/pwg_baker_ckpt_0.4.zip)
Parallel WaveGAN| LJSpeech |[PWGAN-ljspeech](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/ljspeech/voc1)|[pwg_ljspeech_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/pwg_ljspeech_ckpt_0.5.zip)
Parallel WaveGAN| VCTK |[PWGAN-vctk](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/vctk/voc1)|[pwg_vctk_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/pwg_vctk_ckpt_0.5.zip)

### Voice Cloning
Model Type | Dataset| Example Link | Pretrained Models
:-------------:| :------------:| :-----: | :-----
GE2E| AISHELL-3, etc. |[ge2e](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/other/ge2e)|[ge2e_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/ge2e_ckpt_0.3.zip)
GE2E + Tactron2| AISHELL-3 |[ge2e-tactron2-aishell3](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/aishell3/vc0)|[tacotron2_aishell3_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_aishell3_ckpt_0.3.zip)
