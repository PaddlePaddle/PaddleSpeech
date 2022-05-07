
# Released Models

## Speech-to-Text Models

### Speech Recognition Model
Acoustic Model | Training Data | Token-based | Size | Descriptions | CER | WER | Hours of speech | Example Link 
:-------------:| :------------:| :-----: | -----: | :-----: |:-----:| :-----:  | :-----:  | :-----: 
[Ds2 Online Wenetspeech ASR0 Model](https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/asr0_deepspeech2_online_wenetspeech_ckpt_1.0.0a.model.tar.gz) | Wenetspeech Dataset | Char-based | 1.2 GB  | 2 Conv + 5 LSTM layers | 0.152 (test\_net, w/o LM) <br> 0.2417 (test\_meeting, w/o LM) <br> 0.053 (aishell, w/ LM) |-| 10000 h |- 
[Ds2 Online Aishell ASR0 Model](https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_fbank161_ckpt_0.2.1.model.tar.gz) | Aishell Dataset | Char-based | 491 MB  | 2 Conv + 5 LSTM layers | 0.0666 |-| 151 h | [D2 Online Aishell ASR0](../../examples/aishell/asr0) 
[Ds2 Offline Aishell ASR0 Model](https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz)| Aishell Dataset | Char-based | 306 MB | 2 Conv + 3 bidirectional GRU layers| 0.064 |-| 151 h | [Ds2 Offline Aishell ASR0](../../examples/aishell/asr0) 
[Conformer Online Wenetspeech ASR1 Model](https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar.gz) | WenetSpeech Dataset | Char-based | 457 MB  | Encoder:Conformer, Decoder:Transformer, Decoding method: Attention rescoring| 0.11 (test\_net) 0.1879 (test\_meeting) |-| 10000 h |- 
[Conformer Online Aishell ASR1 Model](https://paddlespeech.bj.bcebos.com/s2t/aishell/asr1/asr1_chunk_conformer_aishell_ckpt_0.2.0.model.tar.gz) | Aishell Dataset | Char-based | 189 MB  | Encoder:Conformer, Decoder:Transformer, Decoding method: Attention rescoring| 0.0544 |-| 151 h | [Conformer Online Aishell ASR1](../../examples/aishell/asr1) 
[Conformer Offline Aishell ASR1 Model](https://paddlespeech.bj.bcebos.com/s2t/aishell/asr1/asr1_conformer_aishell_ckpt_0.1.2.model.tar.gz) | Aishell Dataset | Char-based | 189 MB  | Encoder:Conformer, Decoder:Transformer, Decoding method: Attention rescoring | 0.0464 |-| 151 h | [Conformer Offline Aishell ASR1](../../examples/aishell/asr1) 
[Transformer Aishell ASR1 Model](https://paddlespeech.bj.bcebos.com/s2t/aishell/asr1/asr1_transformer_aishell_ckpt_0.1.1.model.tar.gz) | Aishell Dataset | Char-based | 128 MB | Encoder:Transformer, Decoder:Transformer, Decoding method: Attention rescoring | 0.0523 || 151 h | [Transformer  Aishell ASR1](../../examples/aishell/asr1) 
[Ds2 Offline Librispeech ASR0 Model](https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr0/asr0_deepspeech2_librispeech_ckpt_0.1.1.model.tar.gz)| Librispeech Dataset | Char-based | 518 MB | 2 Conv + 3 bidirectional LSTM layers| - |0.0725| 960 h | [Ds2 Offline Librispeech ASR0](../../examples/librispeech/asr0) 
[Conformer Librispeech ASR1 Model](https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz) | Librispeech Dataset | subword-based | 191 MB | Encoder:Conformer, Decoder:Transformer, Decoding method: Attention rescoring |-| 0.0337 | 960 h | [Conformer Librispeech ASR1](../../examples/librispeech/asr1) 
[Transformer Librispeech ASR1 Model](https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_transformer_librispeech_ckpt_0.1.1.model.tar.gz) | Librispeech Dataset | subword-based | 131 MB  | Encoder:Transformer, Decoder:Transformer, Decoding method: Attention rescoring |-| 0.0381 | 960 h | [Transformer Librispeech ASR1](../../examples/librispeech/asr1) 
[Transformer Librispeech ASR2 Model](https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr2/asr2_transformer_librispeech_ckpt_0.1.1.model.tar.gz) | Librispeech Dataset | subword-based | 131 MB  | Encoder:Transformer, Decoder:Transformer, Decoding method: JoinCTC w/ LM |-| 0.0240 | 960 h | [Transformer Librispeech ASR2](../../examples/librispeech/asr2) 

### Language Model based on NGram
Language Model | Training Data | Token-based | Size | Descriptions
:------------:| :------------:|:------------: | :------------: | :------------:
[English LM](https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm) |  [CommonCrawl(en.00)](http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.00.deduped.xz) | Word-based | 8.3 GB | Pruned with 0 1 1 1 1; <br/> About 1.85 billion n-grams; <br/> 'trie'  binary with '-a 22 -q 8 -b 8'
[Mandarin LM Small](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm) | Baidu Internal Corpus | Char-based | 2.8 GB | Pruned with 0 1 2 4 4; <br/> About 0.13 billion n-grams; <br/> 'probing' binary with default settings
[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) | Baidu Internal Corpus | Char-based | 70.4 GB | No Pruning; <br/> About 3.7 billion n-grams; <br/> 'probing' binary with default settings

### Speech Translation Models

| Model | Training Data | Token-based | Size | Descriptions | BLEU | Example Link |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| (only for CLI)[Transformer FAT-ST MTL En-Zh](https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/st1_transformer_mtl_noam_ted-en-zh_ckpt_0.1.1.model.tar.gz) | Ted-En-Zh| Spm| | Encoder:Transformer, Decoder:Transformer, <br />Decoding method: Attention | 20.80 | [Transformer Ted-En-Zh ST1](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/ted_en_zh/st1) |

## Text-to-Speech Models

### Acoustic Models
Model Type | Dataset| Example Link | Pretrained Models|Static Models|Size (static)
:-------------:| :------------:| :-----: | :-----:| :-----:| :-----:
Tacotron2|LJSpeech|[tacotron2-ljspeech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/tts0)|[tacotron2_ljspeech_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/tacotron2/tacotron2_ljspeech_ckpt_0.2.0.zip)|||
Tacotron2|CSMSC|[tacotron2-csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/tts0)|[tacotron2_csmsc_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/tacotron2/tacotron2_csmsc_ckpt_0.2.0.zip)|[tacotron2_csmsc_static_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/tacotron2/tacotron2_csmsc_static_0.2.0.zip)|103MB|
TransformerTTS| LJSpeech| [transformer-ljspeech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/tts1)|[transformer_tts_ljspeech_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/transformer_tts/transformer_tts_ljspeech_ckpt_0.4.zip)|||
SpeedySpeech| CSMSC | [speedyspeech-csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/tts2)|[speedyspeech_csmsc_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/speedyspeech/speedyspeech_csmsc_ckpt_0.2.0.zip)|[speedyspeech_csmsc_static_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/speedyspeech/speedyspeech_csmsc_static_0.2.0.zip)|12MB|
FastSpeech2| CSMSC |[fastspeech2-csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/tts3)|[fastspeech2_nosil_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip)|[fastspeech2_csmsc_static_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_csmsc_static_0.2.0.zip)|157MB|
FastSpeech2-Conformer| CSMSC |[fastspeech2-csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/tts3)|[fastspeech2_conformer_baker_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_conformer_baker_ckpt_0.5.zip)|||
FastSpeech2| AISHELL-3 |[fastspeech2-aishell3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/tts3)|[fastspeech2_nosil_aishell3_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_aishell3_ckpt_0.4.zip)|||
FastSpeech2| LJSpeech |[fastspeech2-ljspeech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/tts3)|[fastspeech2_nosil_ljspeech_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_ljspeech_ckpt_0.5.zip)|||
FastSpeech2| VCTK |[fastspeech2-vctk](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/vctk/tts3)|[fastspeech2_nosil_vctk_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_vctk_ckpt_0.5.zip)|||

### Vocoders
Model Type | Dataset| Example Link | Pretrained Models| Static Models|Size (static)
:-----:| :-----:| :-----: | :-----:| :-----:| :-----:
WaveFlow| LJSpeech |[waveflow-ljspeech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/voc0)|[waveflow_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/waveflow/waveflow_ljspeech_ckpt_0.3.zip)|||
Parallel WaveGAN| CSMSC |[PWGAN-csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc1)|[pwg_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip)|[pwg_baker_static_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_static_0.4.zip)|5.1MB|
Parallel WaveGAN| LJSpeech |[PWGAN-ljspeech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/voc1)|[pwg_ljspeech_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_ljspeech_ckpt_0.5.zip)|||
Parallel WaveGAN| AISHELL-3 |[PWGAN-aishell3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/voc1)|[pwg_aishell3_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip)|||
Parallel WaveGAN| VCTK |[PWGAN-vctk](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/vctk/voc1)|[pwg_vctk_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_vctk_ckpt_0.5.zip)|||
|Multi Band MelGAN | CSMSC |[MB MelGAN-csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc3) | [mb_melgan_csmsc_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_ckpt_0.1.1.zip) <br>[mb_melgan_baker_finetune_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_baker_finetune_ckpt_0.5.zip)|[mb_melgan_csmsc_static_0.1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_static_0.1.1.zip) |8.2MB|
Style MelGAN | CSMSC |[Style MelGAN-csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc4)|[style_melgan_csmsc_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/style_melgan/style_melgan_csmsc_ckpt_0.1.1.zip)| | |
HiFiGAN | CSMSC |[HiFiGAN-csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc5)|[hifigan_csmsc_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_ckpt_0.1.1.zip)|[hifigan_csmsc_static_0.1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_static_0.1.1.zip)|50MB|
HiFiGAN | LJSpeech |[HiFiGAN-ljspeech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/voc5)|[hifigan_ljspeech_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_ljspeech_ckpt_0.2.0.zip)|||
HiFiGAN | AISHELL-3 |[HiFiGAN-aishell3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/voc5)|[hifigan_aishell3_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_ckpt_0.2.0.zip)|||
HiFiGAN | VCTK |[HiFiGAN-vctk](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/vctk/voc5)|[hifigan_vctk_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_vctk_ckpt_0.2.0.zip)|||
WaveRNN | CSMSC |[WaveRNN-csmsc](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc6)|[wavernn_csmsc_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/wavernn/wavernn_csmsc_ckpt_0.2.0.zip)|[wavernn_csmsc_static_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/wavernn/wavernn_csmsc_static_0.2.0.zip)|18MB|


### Voice Cloning
Model Type | Dataset| Example Link | Pretrained Models
:-------------:| :------------:| :-----: | :-----: | 
GE2E| AISHELL-3, etc. |[ge2e](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/ge2e)|[ge2e_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ge2e/ge2e_ckpt_0.3.zip)
GE2E + Tactron2| AISHELL-3 |[ge2e-tactron2-aishell3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/vc0)|[tacotron2_aishell3_ckpt_vc0_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/tacotron2/tacotron2_aishell3_ckpt_vc0_0.2.0.zip)
GE2E + FastSpeech2 | AISHELL-3  |[ge2e-fastspeech2-aishell3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/vc1)|[fastspeech2_nosil_aishell3_vc1_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_aishell3_vc1_ckpt_0.5.zip)


## Audio Classification Models

Model Type | Dataset| Example Link | Pretrained Models | Static Models 
:-------------:| :------------:| :-----: | :-----: | :-----:
PANN | Audioset| [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) | [panns_cnn6.pdparams](https://bj.bcebos.com/paddleaudio/models/panns_cnn6.pdparams), [panns_cnn10.pdparams](https://bj.bcebos.com/paddleaudio/models/panns_cnn10.pdparams), [panns_cnn14.pdparams](https://bj.bcebos.com/paddleaudio/models/panns_cnn14.pdparams) | [panns_cnn6_static.tar.gz](https://paddlespeech.bj.bcebos.com/cls/inference_model/panns_cnn6_static.tar.gz)(18M), [panns_cnn10_static.tar.gz](https://paddlespeech.bj.bcebos.com/cls/inference_model/panns_cnn10_static.tar.gz)(19M), [panns_cnn14_static.tar.gz](https://paddlespeech.bj.bcebos.com/cls/inference_model/panns_cnn14_static.tar.gz)(289M) 
PANN | ESC-50 |[pann-esc50](../../examples/esc50/cls0)|[esc50_cnn6.tar.gz](https://paddlespeech.bj.bcebos.com/cls/esc50/esc50_cnn6.tar.gz), [esc50_cnn10.tar.gz](https://paddlespeech.bj.bcebos.com/cls/esc50/esc50_cnn10.tar.gz), [esc50_cnn14.tar.gz](https://paddlespeech.bj.bcebos.com/cls/esc50/esc50_cnn14.tar.gz)

## Speaker Verification Models

Model Type | Dataset| Example Link | Pretrained Models | Static Models 
:-------------:| :------------:| :-----: | :-----: | :-----:
PANN | VoxCeleb| [voxceleb_ecapatdnn](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/voxceleb/sv0) | [ecapatdnn.tar.gz](https://paddlespeech.bj.bcebos.com/vector/voxceleb/sv0_ecapa_tdnn_voxceleb12_ckpt_0_2_0.tar.gz) | -

## Punctuation Restoration Models
Model Type | Dataset| Example Link | Pretrained Models
:-------------:| :------------:| :-----: | :-----:
Ernie Linear | IWLST2012_zh |[iwslt2012_punc0](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/iwslt2012/punc0)|[ernie_linear_p3_iwslt2012_zh_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/text/ernie_linear_p3_iwslt2012_zh_ckpt_0.1.1.zip)

## Speech Recognition Model  from paddle 1.8

| Acoustic Model |Training Data| Token-based | Size | Descriptions | CER | WER | Hours of speech |
| :-----:| :-----:  |  :-----:  |  :-----:  | :-----:  |  :-----: | :-----:  | :-----: |
| [Ds2 Offline Aishell model](https://deepspeech.bj.bcebos.com/mandarin_models/aishell_model_v1.8_to_v2.x.tar.gz) |        Aishell Dataset  | Char-based  | 234 MB | 2 Conv + 3 bidirectional GRU layers  | 0.0804 | —  | 151 h  |
| [Ds2 Offline Librispeech model](https://deepspeech.bj.bcebos.com/eng_models/librispeech_v1.8_to_v2.x.tar.gz) |      Librispeech Dataset | Word-based  | 307 MB | 2 Conv + 3 bidirectional sharing weight RNN layers | —  | 0.0685 | 960 h |
| [Ds2 Offline Baidu en8k model](https://deepspeech.bj.bcebos.com/eng_models/baidu_en8k_v1.8_to_v2.x.tar.gz) | Baidu Internal English Dataset | Word-based  | 273 MB | 2 Conv + 3 bidirectional GRU layers   |—  | 0.0541 | 8628 h|
