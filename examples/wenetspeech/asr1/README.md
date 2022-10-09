## Pack Model

pack model to tar.gz, e.g.

```bash
./utils/pack_model.sh  --preprocess_conf conf/preprocess.yaml --dict data/vocab.txt conf/conformer.yaml '' data/mean_std.json exp/conformer/checkpoints/wenetspeec
h.pdparams 

```

show model.tar.gz
```
tar tf model.tar.gz 
```

other way is:

```bash
tar cvzf asr1_chunk_conformer_u2_wenetspeech_ckpt_1.1.0.model.tar.gz model.yaml conf/tuning/ conf/chunk_conformer.yaml conf/preprocess.yaml data/mean_std.json exp/chunk_conformer/checkpoints/
```

## Export Static Model

>> Need Paddle >= 2.4

>> `data/test_meeting/data.list`
>> {"input": [{"name": "input1", "shape": [3.2230625, 80], "feat": "/home/PaddleSpeech/dataset/aishell/data_aishell/wav/test/S0764/BAC009S0764W0163.wav", "filetype": "sound"}], "output": [{"name": "target1", "shape": [9, 5538], "text": "\u697c\u5e02\u8c03\u63a7\u5c06\u53bb\u5411\u4f55\u65b9", "token": "\u697c \u5e02 \u8c03 \u63a7 \u5c06 \u53bb \u5411 \u4f55 \u65b9", "tokenid": "1891 1121 3502 1543 1018 477 528 163 1657"}], "utt": "BAC009S0764W0163", "utt2spk": "S0764"}

>> Test Wav: 
>> wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
### U2 chunk conformer
>> UiDecoder
>> Make sure `reverse_weight` in config is `0.0`
>> https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_u2_wenetspeech_ckpt_1.1.0.model.tar.gz
```
tar zxvf asr1_chunk_conformer_u2_wenetspeech_ckpt_1.1.0.model.tar.gz
./local/export.sh conf/chunk_conformer.yaml exp/chunk_conformer/checkpoints/avg_10 ./export.ji
```

### U2++ chunk conformer
>> BiDecoder
>> https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_u2pp_wenetspeech_ckpt_1.1.0.model.tar.gz
>> Make sure `reverse_weight` in config is not `0.0`

```
./local/export.sh conf/chunk_conformer_u2pp.yaml exp/chunk_conformer/checkpoints/avg_10 ./export.ji
```
