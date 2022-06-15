# DeepSpeech2 ONNX model

1. convert deepspeech2 model to ONNX, using Paddle2ONNX.
2. check paddleinference and onnxruntime output equal.
3. optimize onnx model
4. check paddleinference and optimized onnxruntime output equal.

Please make sure [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) and [onnx-simplifier](https://github.com/zh794390558/onnx-simplifier/tree/dyn_time_shape) version is correct.

The example test with these packages installed:
```
paddle2onnx              0.9.8rc0    # develop af4354b4e9a61a93be6490640059a02a4499bc7a
paddleaudio              0.2.1
paddlefsl                1.1.0
paddlenlp                2.2.6
paddlepaddle-gpu         2.2.2
paddlespeech             0.0.0       # develop
paddlespeech-ctcdecoders 0.2.0
paddlespeech-feat        0.1.0
onnx                     1.11.0
onnx-simplifier          0.0.0       # https://github.com/zh794390558/onnx-simplifier/tree/dyn_time_shape
onnxoptimizer            0.2.7
onnxruntime              1.11.0
```

## Using

```
bash run.sh
```

For more details please see `run.sh`.

## Outputs
The optimized onnx model is `exp/model.opt.onnx`.

To show the graph, please using `local/netron.sh`.
