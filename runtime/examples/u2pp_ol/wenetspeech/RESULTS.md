# aishell test

7176 utts, duration 36108.9 sec.

## U2++ Attention Rescore

> Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz, support `avx512_vnni`
> RTF with feature and decoder which is more end to end.

### FP32

`local/recognizer.sh`

#### CER

```
Overall -> 5.75 % N=104765 C=99035 S=5587 D=143 I=294
Mandarin -> 5.75 % N=104762 C=99035 S=5584 D=143 I=294
English -> 0.00 % N=0 C=0 S=0 D=0 I=0
Other -> 100.00 % N=3 C=0 S=3 D=0 I=0
```

#### RTF 

```
I1027 10:52:38.662868 51665 recognizer_main.cc:122] total wav duration is: 36108.9 sec
I1027 10:52:38.662858 51665 recognizer_main.cc:121] total cost:9577.31 sec
I1027 10:52:38.662876 51665 recognizer_main.cc:123] RTF is: 0.265234
```

### INT8

`local/recognizer_quant.sh`

#### CER

```
Overall -> 5.83 % N=104765 C=98943 S=5675 D=147 I=286
Mandarin -> 5.83 % N=104762 C=98943 S=5672 D=147 I=286
English -> 0.00 % N=0 C=0 S=0 D=0 I=0
Other -> 100.00 % N=3 C=0 S=3 D=0 I=0
```

#### RTF 

```
I1110 09:59:52.551712 37249 u2_recognizer_main.cc:122] total wav duration is: 36108.9 sec
I1110 09:59:52.551717 37249 u2_recognizer_main.cc:123] total decode cost:9737.63 sec
I1110 09:59:52.551723 37249 u2_recognizer_main.cc:124] RTF is: 0.269674
```

### TLG decoder without attention rescore

`local/recognizer_wfst.sh`

#### CER

```
Overall -> 4.73 % N=104765 C=100001 S=4283 D=481 I=187
Mandarin -> 4.72 % N=104762 C=100001 S=4280 D=481 I=187
Other -> 100.00 % N=3 C=0 S=3 D=0 I=0
```

#### RTF
```
I0417 08:07:15.300631 75784 recognizer_main.cc:113] total wav duration is: 36108.9 sec
I0417 08:07:15.300642 75784 recognizer_main.cc:114] total decode cost:10247.7 sec
I0417 08:07:15.300648 75784 recognizer_main.cc:115] total rescore cost:908.228 sec
I0417 08:07:15.300653 75784 recognizer_main.cc:116] RTF is: 0.283
```
