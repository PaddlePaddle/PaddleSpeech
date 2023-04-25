# U2/U2++ Streaming ASR 

A C++ deployment example for `PaddleSpeech/examples/wenetspeech/asr1` recipe. The model is static model from `export`, how to export model please see [here](../../../../examples/wenetspeech/asr1/). If you want using exported model, `run.sh` will download it, for the model link please see `run.sh`.

This example will demonstrate how to using the u2/u2++ model to recognize `wav` and compute `CER`. We using AISHELL-1 as test data.

## Testing with Aishell Test Data

## Source path.sh

```bash
. path.sh
```

SpeechX bins is under `echo $SPEECHX_BUILD`, more info please see `path.sh`.


### Download dataset and model

```
./run.sh --stop_stage 0
```

### process `cmvn` and compute feature

```bash
./run.sh --stage 1 --stop_stage 1
```

If you only want to convert `cmvn` file format, can using this cmd:

```bash 
./local/feat.sh --stage 1 --stop_stage 1
```

### Decoding using `feature` input

```
./run.sh --stage 2 --stop_stage 2
```

### Decoding using `wav` input

```
./run.sh --stage 3 --stop_stage 3
```

This stage using `u2_recognizer_main` to recognize wav file.

The input is `scp` file which look like this:
```text
# head data/split1/1/aishell_test.scp 
BAC009S0764W0121        /workspace/PaddleSpeech/runtime/examples/u2pp_ol/wenetspeech/data/test/S0764/BAC009S0764W0121.wav
BAC009S0764W0122        /workspace/PaddleSpeech/runtime/examples/u2pp_ol/wenetspeech/data/test/S0764/BAC009S0764W0122.wav
...
BAC009S0764W0125        /workspace/PaddleSpeech/runtime/examples/u2pp_ol/wenetspeech/data/test/S0764/BAC009S0764W0125.wav
```

If you want to recognize one wav, you can make `scp` file like this:
```text
key  path/to/wav/file
```

Then specify `--wav_rspecifier=` param for `u2_recognizer_main` bin. For other flags meaning, please see `help`:
```bash
u2_recognizer_main --help
```

The exmaple using `u2_recgonize_main` bin please see `local/recognizer.sh`.

### Decoding with `wav` using quant model

`local/recognizer_quant.sh` is same to `local/recognizer.sh`, but using quanted model.


## Results

Please see [here](./RESULTS.md).
