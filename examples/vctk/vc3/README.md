You can download test source audios from [test_wav.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/starganv2vc/test_wav.zip).


Test Voice Conversion:

```bash
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/starganv2vc/test_wav.zip
unzip test_wav.zip
./run.sh --stage 2 --stop-stage 2 --gpus 0
```