
# Trying Live Demo with Your Own Voice

Until now, an ASR model is trained and tested qualitatively (`infer`) and quantitatively (`test`) with existing audio files. But it is not yet tested with your own speech. We build up a real-time demo ASR engine with the trained model, enabling you to test and play around with the demo, with your own voice.

First, change your directory to `examples/aishell` and `source path.sh`.

To start the demo's server, please run this in one console:

```bash
CUDA_VISIBLE_DEVICES=0 bash local/server.sh
```

For the machine (might not be the same machine) to run the demo's client, please do the following installation before moving on.

For example, on MAC OS X:

```bash
brew install portaudio
pip install pyaudio
pip install keyboard
```

Then to start the client, please run this in another console:

```bash
CUDA_VISIBLE_DEVICES=0 bash local/client.sh
```

Now, in the client console, press the `whitespace` key, hold, and start speaking. Until finishing your utterance, release the key to let the speech-to-text results shown in the console. To quit the client, just press `ESC` key.

Notice that `deepspeech/exps/deepspeech2/deploy/client.py` must be run on a machine with a microphone device, while `deepspeech/exps/deepspeech2/deploy/server.py` could be run on one without any audio recording hardware, e.g. any remote server machine. Just be careful to set the `host_ip` and `host_port` argument with the actual accessible IP address and port, if the server and client are running with two separate machines. Nothing should be done if they are running on one single machine.

Please also refer to `examples/aishell/local/server.sh`, which will first download a pre-trained Chinese model (trained with AISHELL1) and then start the demo server with the model. With running `examples/aishell/local/client.sh`, you can speak Chinese to test it. If you would like to try some other models, just update `--checkpoint_path` argument in the script.  
