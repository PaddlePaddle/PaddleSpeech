# Examples for SpeechX

* ds2_ol - ds2 streaming test under `aishell-1` test dataset.  
The entrypoint is `ds2_ol/aishell/run.sh`


## How to run  

`run.sh` is the entry point.

Example to play `ds2_ol`:

```
pushd ds2_ol/aishell
bash run.sh
```

## Display Model with [Netron](https://github.com/lutzroeder/netron)  

```
pip install netron
netron exp/deepspeech2_online/checkpoints/avg_1.jit.pdmodel  --port 8022 --host 10.21.55.20
```

## For Developer  

> Warning: Only for developer, make sure you know what's it.

* dev - for speechx developer, using for test.

## Build WFST  

> Warning: Using below example when you know what's it.

* text_lm - process text for build lm
* ngram - using to build NGram ARPA lm.
* wfst - build wfst for TLG.
