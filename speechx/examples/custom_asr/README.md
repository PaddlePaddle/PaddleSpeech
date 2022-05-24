# customized Auto Speech Recognition

## introduction
These scripts are tutorials to show you how build your own decoding graph.

eg:
* G with slot: 打车到 "address_slot"。
![](https://ai-studio-static-online.cdn.bcebos.com/28d9ef132a7f47a895a65ae9e5c4f55b8f472c9f3dd24be8a2e66e0b88b173a4)

* this is address slot wfst, you can add the address which want to recognize.
![](https://ai-studio-static-online.cdn.bcebos.com/47c89100ef8c465bac733605ffc53d76abefba33d62f4d818d351f8cea3c8fe2)

* after replace operation, G = fstreplace(G_with_slot, address_slot), we will get the customized graph.
![](https://ai-studio-static-online.cdn.bcebos.com/60a3095293044f10b73039ab10c7950d139a6717580a44a3ba878c6e74de402b)

These operations are in the scripts, please check out. we will lanuch more detail scripts.

## How to run

```
bash run.sh
```

## Results

### CTC WFST

```
Overall -> 1.23 % N=1134 C=1126 S=6 D=2 I=6
Mandarin -> 1.24 % N=1132 C=1124 S=6 D=2 I=6
English -> 0.00 % N=2 C=2 S=0 D=0 I=0
```
