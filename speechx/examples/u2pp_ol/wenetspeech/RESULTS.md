# aishell test

7176 utts, duration 36108.9 sec.

## U2++ Attention Rescore

> Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz, support `avx512_vnni`
> RTF with feature and decoder which is more end to end.
### FP32

#### CER

```
Overall -> 5.75 % N=104765 C=99035 S=5587 D=143 I=294
Mandarin -> 5.75 % N=104762 C=99035 S=5584 D=143 I=294
English -> 0.00 % N=0 C=0 S=0 D=0 I=0
Other -> 100.00 % N=3 C=0 S=3 D=0 I=0
```

#### RTF 

```
I1027 10:52:38.662868 51665 u2_recognizer_main.cc:122] total wav duration is: 36108.9 sec
I1027 10:52:38.662858 51665 u2_recognizer_main.cc:121] total cost:11169.1 sec
I1027 10:52:38.662876 51665 u2_recognizer_main.cc:123] RTF is: 0.309318
```

### INT8

> RTF relative improve 12.8%, which count feature and decoder time.
> Test under Paddle commit c331e2ce2031d68a553bc9469a07c30d718438f3  

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
