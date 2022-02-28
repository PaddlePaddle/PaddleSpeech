# Speaker recognition experiments with VoxCeleb.
This folder contains scripts for running speaker identification and verification experiments with the VoxCeleb dataset(http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

# Training Xvectors
Run the following command to train xvectors:

`python train_speaker_embeddings.py hparams/train_x_vectors.yaml`

You can use the same script for voxceleb1, voxceleb2, and voxceleb1+2. Just change the datafolder and the corresponding number of speakers (1211 vox1, 5994 vox2, 7205 vox1+2).
For voxceleb1 + voxceleb2, see preparation instructions below).

The system trains a TDNN for speaker embeddings coupled with a speaker-id classifier. The speaker-id accuracy should be around 97-98% for both voxceleb1 and voceleb2.

# Speaker verification with PLDA
After training the speaker embeddings, it is possible to perform speaker verification using PLDA.  You can run it with the following command:

`python speaker_verification_plda.py hparams/verification_plda_xvector.yaml`

If you didn't train the speaker embedding before, we automatically download the xvector model from the web.
This system achieves an EER = 3.2% on voxceleb1 + voxceleb2.
These results are all obtained with the official verification split of voxceleb1 (veri\_split.txt)


# Speaker verification using ECAPA-TDNN embeddings
Run the following command to train speaker embeddings using [ECAPA-TDNN](https://arxiv.org/abs/2005.07143):

`python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml`


The speaker-id accuracy should be around 98-99% for both voxceleb1 and voceleb2.

After training the speaker embeddings, it is possible to perform speaker verification using cosine similarity.  You can run it with the following command:

`python speaker_verification_cosine.py hparams/verification_ecapa.yaml`

This system achieves an EER = 0.69 % on voxceleb1 + voxceleb2.
These results are all obtained with the official verification split of voxceleb1 (veri\_split.txt)

# VoxCeleb2 preparation
Voxceleb2 audio files are released in m4a format. All the files must be converted in wav files before
feeding them is SpeechBrain. Please, follow these steps to prepare the dataset correctly:

1. Download both Voxceleb1 and Voxceleb2.
You can find download instructions here: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
Note that for the speaker verification experiments with Voxceleb2 the official split of voxceleb1 is used to compute EER.

2. Convert .m4a to wav
Voxceleb2 stores files with the m4a audio format. To use them within SpeechBrain you have to convert all the m4a files into wav files.
You can do the conversion using ffmpeg(see for instance conversion scripts in https://gitmemory.com/issue/pytorch/audio/104/493137979 or https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830). This operation might take several hours and should be only once.

2. Put all the wav files in a folder called wav. You should have something like `voxceleb2/wav/id*/*.wav` (e.g, `voxceleb2/wav/id00012/21Uxsk56VDQ/00001.wav`)

3. copy the `voxceleb1/vox1_test_wav.zip` file into the voxceleb2 folder.

4. Unpack voxceleb1 test files(verification split).

Go to the voxceleb2 folder and run `unzip vox1_test_wav.zip`.

5. Copy the verification split(`voxceleb1/ meta/veri_test.txt`) into voxceleb2(`voxceleb2/meta/ veri_test.txt`)

6. Now everything is ready and you can run voxceleb2 experiments:
- training embeddings:

`python train_speaker_embeddings.py hparams/train_xvectors.yaml`

Note: To prepare the voxceleb1 + voxceleb2 dataset you have to copy and unpack vox1_dev_wav.zip for the voxceleb1 dataset.

# Performance summary

[Speaker Verification Results with Voxceleb 1 + Voxceleb2]
| System          | Dataset    | EER  | Model/Log Link |
|-----------------|------------|------| -----|
| Xvector + PLDA  | VoxCeleb 1,2 | 3.2% | https://drive.google.com/drive/folders/1TLKByLRkgkUiDV2coMrIh-OMHANrnOl-?usp=sharing |
| ECAPA-TDNN      | Voxceleb 1,2 | 0.69% | https://drive.google.com/file/d/1EziERcHD_gyE6qc8DbxPKU1isVf7pbNl/view?usp=sharing|


# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```


