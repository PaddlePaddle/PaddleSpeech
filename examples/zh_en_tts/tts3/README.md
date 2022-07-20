# Test
We train a Chinese-English mixed fastspeech2 model. The training code is still being sorted out, let's show how to use it first.
The sample rate of the synthesized audio is 22050 Hz. 

## Download pretrained models
Put pretrained models in a directory named `models`.

- [fastspeech2_csmscljspeech_add-zhen.zip](https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_csmscljspeech_add-zhen.zip)
- [hifigan_ljspeech_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_ljspeech_ckpt_0.2.0.zip)

```bash
mkdir models
cd models
wget https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_csmscljspeech_add-zhen.zip
unzip fastspeech2_csmscljspeech_add-zhen.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_ljspeech_ckpt_0.2.0.zip
unzip hifigan_ljspeech_ckpt_0.2.0.zip
cd ../
```

## test
You can choose `--spk_id` {0, 1} in `local/synthesize_e2e.sh`.

```bash
bash test.sh
```
