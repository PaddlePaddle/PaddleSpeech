# Aishell - Deepspeech2 Streaming

## CTC Prefix Beam Search w/o LM

```
Overall -> 16.14 % N=104612 C=88190 S=16110 D=312 I=465
Mandarin -> 16.14 % N=104612 C=88190 S=16110 D=312 I=465
Other -> 0.00 % N=0 C=0 S=0 D=0 I=0
```

## CTC Prefix Beam Search w LM

LM: zh_giga.no_cna_cmn.prune01244.klm
```
Overall -> 7.86 % N=104768 C=96865 S=7573 D=330 I=327
Mandarin -> 7.86 % N=104768 C=96865 S=7573 D=330 I=327
Other -> 0.00 % N=0 C=0 S=0 D=0 I=0
```

## CTC WFST

LM: aishell train
```
Overall -> 11.14 % N=103017 C=93363 S=9583 D=71 I=1819
Mandarin -> 11.14 % N=103017 C=93363 S=9583 D=71 I=1818
Other -> 0.00 % N=0 C=0 S=0 D=0 I=1
```