# Tacotron2 + AISHELL-3 Voice Cloning
This example contains code used to train a [Tacotron2 ](https://arxiv.org/abs/1712.05884) model with [AISHELL-3](http://www.aishelltech.com/aishell_3). The trained model can be used in Voice Cloning Task, We refer to the model structure of  [Transfer Learning from Speaker Veriﬁcation to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) . The general steps are as follows:
1. Speaker Encoder: We  use a Speaker Verification to train a speaker encoder. Datasets used in this task are different from those used in Tacotron2, because the  transcriptions are not needed, we use more datasets, refer to  [ge2e](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/other/ge2e).
2. Synthesizer: Then, we use the trained speaker encoder to generate utterance embedding for each  sentence in AISHELL-3. This embedding is a extra input of  Tacotron2 which will be concated with encoder outputs.
3. Vocoder: We use WaveFlow as the neural Vocoder, refer to [waveflow](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/ljspeech/voc0).

## Get Started
Assume the path to the dataset is `~/datasets/data_aishell3`.
Assume the path to the MFA result of AISHELL-3 is `./alignment`.
Assume the path to the pretrained ge2e model is `ge2e_ckpt_path=./ge2e_ckpt_0.3/step-3000000`
Run the command below to
1. **source path**.
2. preprocess the dataset,
3. train the model.
4. start a voice cloning inference.
```bash
./run.sh
```
### Preprocess the dataset
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/preprocess.sh ${input} ${preprocess_path} ${alignment} ${ge2e_ckpt_path}
```
#### generate utterance embedding
 Use pretrained GE2E (speaker encoder) to generate utterance embedding for each sentence in AISHELL-3, which has the same file structure with wav files and the format is  `.npy`.

```bash
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../ge2e/inference.py \
        --input=${input} \
        --output=${preprocess_path}/embed \
        --device="gpu" \
        --checkpoint_path=${ge2e_ckpt_path}
fi
```

The computing time of  utterance embedding can be x hours.
####  process wav
There are silence in the edge of AISHELL-3's wavs, and the audio amplitude is very small, so, we need to remove the silence and normalize the audio. You can the silence remove method based on   volume or energy, but the effect is not very good, We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get  the alignment of text and  speech, then utilize the alignment results to remove the silence.

We use Montreal Force Aligner 1.0. The label in  aishell3 include pinyin，so the lexicon we provided to MFA is pinyin rather than Chinese characters. And the prosody marks(`$`  and `%`) need to be removed. You shoud preprocess the dataset into the format  which MFA needs, the texts have the same name with wavs and have the suffix `.lab`.

We use [lexicon.txt](https://github.com/PaddlePaddle/DeepSpeech/blob/develop/parakeet/exps/voice_cloning/tacotron2_ge2e/lexicon.txt) as the lexicon.

You can download the alignment results from here [alignment_aishell3.tar.gz](https://paddlespeech.bj.bcebos.com/Parakeet/alignment_aishell3.tar.gz), or train your own MFA model reference to [use_mfa example](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/other/use_mfa) (use MFA1.x now) of our repo.

```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Process wav ..."
    python3 ${BIN_DIR}/process_wav.py \
        --input=${input}/wav \
        --output=${preprocess_path}/normalized_wav \
        --alignment=${alignment}
fi
```

#### preprocess transcription
We revert the transcription into `phones` and  `tones`. It is worth noting that our processing here is different from that used for MFA, we separated the tones. This is a processing method, of course, you can only segment initials and vowels.

```bash
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python3 ${BIN_DIR}/preprocess_transcription.py \
        --input=${input} \
        --output=${preprocess_path}
fi
```
The default input is  `~/datasets/data_aishell3/train`，which contains `label_train-set.txt`, the processed results are `metadata.yaml` and  `metadata.pickle`. the former is a text format for easy viewing, and the latter is a binary format for direct reading.
#### extract mel
```python
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python3 ${BIN_DIR}/extract_mel.py \
        --input=${preprocess_path}/normalized_wav \
        --output=${preprocess_path}/mel
fi
```

###  Train the model
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${preprocess_path} ${train_output_path}
```

Our model remve  stop token prediction in Tacotron2, because of the problem of extremely unbalanced proportion of positive and negative samples of stop token prediction, and it's very sensitive to the clip of audio silence. We use the last symbol from the highest point of attention to the encoder side as the termination condition.

In addition, in order to accelerate the convergence of the model, we add `guided attention loss` to induce the alignment between encoder and decoder to show diagonal lines faster.
###  Infernece
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/voice_cloning.sh ${ge2e_params_path} ${tacotron2_params_path} ${waveflow_params_path} ${vc_input} ${vc_output}
```
## Pretrained Model
[tacotron2_aishell3_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_aishell3_ckpt_0.3.zip).
