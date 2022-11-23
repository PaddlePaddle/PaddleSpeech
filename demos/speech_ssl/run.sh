#!/bin/bash

# audio download
<<<<<<< HEAD
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
=======
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
>>>>>>> 45426846942f68cf43a23677d8d55f6d4ab93ab1

# to recognize text 
paddlespeech ssl --task asr --lang en --input ./en.wav

# to get acoustic representation
paddlespeech ssl --task vector --lang en --input ./en.wav
<<<<<<< HEAD

=======
>>>>>>> 45426846942f68cf43a23677d8d55f6d4ab93ab1
