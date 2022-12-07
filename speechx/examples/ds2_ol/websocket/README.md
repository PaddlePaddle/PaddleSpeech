#  Streaming DeepSpeech2 Server with WebSocket

This example is about using `websocket` as streaming deepspeech2 server. For deepspeech2 model training please see [here](../../../../examples/aishell/asr0/).

The websocket protocal is same to [PaddleSpeech Server](../../../../demos/streaming_asr_server/), 
for detail of implementation please see [here](../../../speechx/protocol/websocket/).


## Source path.sh

```bash
. path.sh
```

SpeechX bins is under `echo $SPEECHX_BUILD`, more info please see `path.sh`.


## Start WebSocket Server

```bash
bash websoket_server.sh
```

The output is like below:

```text
I1130 02:19:32.029882 12856 cmvn_json2kaldi_main.cc:39] cmvn josn path: /workspace/zhanghui/PaddleSpeech/speechx/examples/ds2_ol/websocket/data/model/data/mean_std.json
I1130 02:19:32.032230 12856 cmvn_json2kaldi_main.cc:73] nframe: 907497
I1130 02:19:32.032564 12856 cmvn_json2kaldi_main.cc:85] cmvn stats have write into: /workspace/zhanghui/PaddleSpeech/speechx/examples/ds2_ol/websocket/data/cmvn.ark
I1130 02:19:32.032579 12856 cmvn_json2kaldi_main.cc:86] Binary: 1
I1130 02:19:32.798342 12937 feature_pipeline.h:53] cmvn file: /workspace/zhanghui/PaddleSpeech/speechx/examples/ds2_ol/websocket/data/cmvn.ark
I1130 02:19:32.798542 12937 feature_pipeline.h:58] dither: 0
I1130 02:19:32.798583 12937 feature_pipeline.h:60] frame shift ms: 10
I1130 02:19:32.798588 12937 feature_pipeline.h:62] feature type: linear
I1130 02:19:32.798596 12937 feature_pipeline.h:80] frame length ms: 20
I1130 02:19:32.798601 12937 feature_pipeline.h:88] subsampling rate: 4
I1130 02:19:32.798606 12937 feature_pipeline.h:90] nnet receptive filed length: 7
I1130 02:19:32.798611 12937 feature_pipeline.h:92] nnet chunk size: 1
I1130 02:19:32.798615 12937 feature_pipeline.h:94] frontend fill zeros: 0
I1130 02:19:32.798630 12937 nnet_itf.h:52] subsampling rate: 4
I1130 02:19:32.798635 12937 nnet_itf.h:54] model path: /workspace/zhanghui/PaddleSpeech/speechx/examples/ds2_ol/websocket/data/model/exp/deepspeech2_online/checkpoints//avg_1.jit.pdmodel
I1130 02:19:32.798640 12937 nnet_itf.h:57] param path: /workspace/zhanghui/PaddleSpeech/speechx/examples/ds2_ol/websocket/data/model/exp/deepspeech2_online/checkpoints//avg_1.jit.pdiparams
I1130 02:19:32.798643 12937 nnet_itf.h:59] DS2 param: 
I1130 02:19:32.798647 12937 nnet_itf.h:61]   cache names: chunk_state_h_box,chunk_state_c_box
I1130 02:19:32.798652 12937 nnet_itf.h:63]   cache shape: 5-1-1024,5-1-1024
I1130 02:19:32.798656 12937 nnet_itf.h:65]   input names: audio_chunk,audio_chunk_lens,chunk_state_h_box,chunk_state_c_box
I1130 02:19:32.798660 12937 nnet_itf.h:67]   output names: softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0
I1130 02:19:32.798664 12937 ctc_tlg_decoder.h:41] fst path: /workspace/zhanghui/PaddleSpeech/speechx/examples/ds2_ol/websocket/data/wfst//TLG.fst
I1130 02:19:32.798669 12937 ctc_tlg_decoder.h:42] fst symbole table: /workspace/zhanghui/PaddleSpeech/speechx/examples/ds2_ol/websocket/data/wfst//words.txt
I1130 02:19:32.798673 12937 ctc_tlg_decoder.h:47] LatticeFasterDecoder max active: 7500
I1130 02:19:32.798677 12937 ctc_tlg_decoder.h:49] LatticeFasterDecoder beam: 15
I1130 02:19:32.798681 12937 ctc_tlg_decoder.h:50] LatticeFasterDecoder lattice_beam: 7.5
I1130 02:19:32.798708 12937 websocket_server_main.cc:37] Listening at port 8082
```

## Start WebSocket Client

```bash
bash websocket_client.sh
```

This script using AISHELL-1 test data to call websocket server.

The input is specific by `--wav_rspecifier=scp:$data/$aishell_wav_scp`.

The `scp` file which look like this:
```text
# head data/split1/1/aishell_test.scp 
BAC009S0764W0121        /workspace/PaddleSpeech/speechx/examples/u2pp_ol/wenetspeech/data/test/S0764/BAC009S0764W0121.wav
BAC009S0764W0122        /workspace/PaddleSpeech/speechx/examples/u2pp_ol/wenetspeech/data/test/S0764/BAC009S0764W0122.wav
...
BAC009S0764W0125        /workspace/PaddleSpeech/speechx/examples/u2pp_ol/wenetspeech/data/test/S0764/BAC009S0764W0125.wav
```

If you want to recognize one wav, you can make `scp` file like this:
```text
key  path/to/wav/file
```
