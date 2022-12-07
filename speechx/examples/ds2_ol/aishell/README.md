# Aishell - Deepspeech2 Streaming

> We recommend using U2/U2++ model instead of DS2, please see [here](../../u2pp_ol/wenetspeech/).

A C++ deployment example for using the deepspeech2 model to recognize `wav` and compute `CER`. We using AISHELL-1 as test data.

## Source path.sh

```bash
. path.sh
```

SpeechX bins is under `echo $SPEECHX_BUILD`, more info please see `path.sh`.

## Recognize with linear feature

```bash
bash run.sh
```

`run.sh` has multi stage, for details please see `run.sh`: 

1. donwload dataset, model and lm
2. convert cmvn format and compute feature
3. decode w/o lm by feature
4. decode w/ ngram lm by feature
5. decode w/ TLG graph by feature
6. recognize w/ TLG graph by wav input

### Recognize with `.scp` file for wav

This sciprt using `recognizer_main` to recognize wav file.

The input is `scp` file which look like this:
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

Then specify `--wav_rspecifier=` param for `recognizer_main` bin. For other flags meaning, please see `help`:
```bash
recognizer_main --help
```

For the exmaple to using `recognizer_main` please see `run.sh`.


### CTC Prefix Beam Search w/o LM

```
Overall -> 16.14 % N=104612 C=88190 S=16110 D=312 I=465
Mandarin -> 16.14 % N=104612 C=88190 S=16110 D=312 I=465
Other -> 0.00 % N=0 C=0 S=0 D=0 I=0
```

### CTC Prefix Beam Search w/ LM

LM: zh_giga.no_cna_cmn.prune01244.klm
```
Overall -> 7.86 % N=104768 C=96865 S=7573 D=330 I=327
Mandarin -> 7.86 % N=104768 C=96865 S=7573 D=330 I=327
Other -> 0.00 % N=0 C=0 S=0 D=0 I=0
```

### CTC TLG WFST

LM: [aishell train](http://paddlespeech.bj.bcebos.com/speechx/examples/ds2_ol/aishell/aishell_graph.zip)
--acoustic_scale=1.2
```
Overall -> 11.14 % N=103017 C=93363 S=9583 D=71 I=1819
Mandarin -> 11.14 % N=103017 C=93363 S=9583 D=71 I=1818
Other -> 0.00 % N=0 C=0 S=0 D=0 I=1
```

LM: [wenetspeech](http://paddlespeech.bj.bcebos.com/speechx/examples/ds2_ol/aishell/wenetspeech_graph.zip)
--acoustic_scale=1.5
```
Overall -> 10.93 % N=104765 C=93410 S=9780 D=1575 I=95
Mandarin -> 10.93 % N=104762 C=93410 S=9779 D=1573 I=95
Other -> 100.00 % N=3 C=0 S=1 D=2 I=0
```

## Recognize with fbank feature

This script is same to `run.sh`, but using fbank feature.

```bash
bash run_fbank.sh
```

### CTC Prefix Beam Search w/o LM

```
Overall -> 10.44 % N=104765 C=94194 S=10174 D=397 I=369
Mandarin -> 10.44 % N=104762 C=94194 S=10171 D=397 I=369
Other -> 100.00 % N=3 C=0 S=3 D=0 I=0
```

### CTC Prefix Beam Search w/ LM

LM: zh_giga.no_cna_cmn.prune01244.klm

```
Overall -> 5.82 % N=104765 C=99386 S=4944 D=435 I=720
Mandarin -> 5.82 % N=104762 C=99386 S=4941 D=435 I=720
English -> 0.00 % N=0 C=0 S=0 D=0 I=0
```

### CTC TLG WFST

LM: [aishell train](https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_graph2.zip)
```
Overall -> 9.58 % N=104765 C=94817 S=4326 D=5622 I=84
Mandarin -> 9.57 % N=104762 C=94817 S=4325 D=5620 I=84
Other -> 100.00 % N=3 C=0 S=1 D=2 I=0
```

## Build TLG WFST graph 

The script is for building TLG wfst graph, depending on `srilm`, please make sure it is installed.
For more information please see the script below.

```bash
 bash ./local/run_build_tlg.sh
```
