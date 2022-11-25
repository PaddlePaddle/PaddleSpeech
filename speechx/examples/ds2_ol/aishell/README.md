# Aishell - Deepspeech2 Streaming

## tutorial 

## how to use in user's project

> Users want to learn how to use the example for their project ! Not just test standard result ! If users want to use these codes in their project, how to use our code ? For example, there are some audios that needs to be recognized, how to get the result . Please show it in README

> What are the requirements for audio format ? 

> if there are some bins or exes builded by speechx（such as `recognizer_main`）, how to find the enterpoint source code ? how to use these program ? What are the meanings of these parameters ? What are the limits of these parameters ? Please show it in README, not just write it in run.sh .


## How to run

```
bash run.sh
```

## Results

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

### CTC WFST

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

## fbank
```
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

### CTC WFST

LM: [aishell train](https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_graph2.zip)
```
Overall -> 9.58 % N=104765 C=94817 S=4326 D=5622 I=84
Mandarin -> 9.57 % N=104762 C=94817 S=4325 D=5620 I=84
Other -> 100.00 % N=3 C=0 S=1 D=2 I=0
```

## build TLG graph 
```
 bash run_build_tlg.sh
```
