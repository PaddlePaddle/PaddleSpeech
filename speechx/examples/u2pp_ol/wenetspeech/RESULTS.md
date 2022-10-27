# aishell test

7176 utts, duration 36108.9 sec.

## Attention Rescore

### u2++ FP32

#### CER

```
Overall -> 5.75 % N=104765 C=99035 S=5587 D=143 I=294
Mandarin -> 5.75 % N=104762 C=99035 S=5584 D=143 I=294
English -> 0.00 % N=0 C=0 S=0 D=0 I=0
Other -> 100.00 % N=3 C=0 S=3 D=0 I=0
```

#### RTF 

> RTF with feature and decoder which is more end to end.

* Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz, support `avx512_vnni`

```
I1027 10:52:38.662868 51665 u2_recognizer_main.cc:122] total wav duration is: 36108.9 sec
I1027 10:52:38.662858 51665 u2_recognizer_main.cc:121] total cost:11169.1 sec
I1027 10:52:38.662876 51665 u2_recognizer_main.cc:123] RTF is: 0.309318
```

* Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz, not support `avx512_vnni`

```
I1026 16:13:26.247121 48038 u2_recognizer_main.cc:123] total wav duration is: 36108.9 sec
I1026 16:13:26.247130 48038 u2_recognizer_main.cc:124] total decode cost:13656.7 sec
I1026 16:13:26.247138 48038 u2_recognizer_main.cc:125] RTF is: 0.378208
```
