# Examples for SpeechX

> `u2pp_ol` is recommended.

* `u2pp_ol` - u2++ streaming asr test under `aishell-1` test dataset.
* `ds2_ol` - ds2 streaming test under `aishell-1` test dataset. 


## How to run  

### Create env

Using `tools/evn.sh` under `speechx` to create python env.

```
bash tools/env.sh
```

Source env before play with example.
```
. venv/bin/activate
```

### Play with example

`run.sh` is the entry point for every example.

Example to play `u2pp_ol`:

```
pushd u2pp_ol/wenetspeech
bash run.sh --stop_stage 4
```

## Display Model with [Netron](https://github.com/lutzroeder/netron)  

If you have a model, we can using this commnd to show model graph.

For example:
```
pip install netron
netron exp/deepspeech2_online/checkpoints/avg_1.jit.pdmodel  --port 8022 --host 10.21.55.20
```

## For Developer  

> Reminder: Only for developer, make sure you know what's it.

* codelab - for speechx developer, using for test.
