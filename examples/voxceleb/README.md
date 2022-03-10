
dataset info refer to [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#about)

sv0 - speaker verfication with softmax backend etc, all python code
      more info refer to the sv0/readme.txt

sv1 - dependence on kaldi, speaker verfication with plda/sc backend, 
      more info refer to the sv1/readme.txt


## VoxCeleb2 preparation

VoxCeleb2 audio files are released in m4a format. All the VoxCeleb2 m4a audio files must be converted in wav files before feeding them in PaddleSpeech. 
Please, follow these steps to prepare the dataset correctly:

1. Download Voxceleb2.
You can find download instructions here: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

2. Convert .m4a to wav
VoxCeleb2 stores files with the m4a audio format. To use them in PaddleSpeech,  you have to convert all the m4a audio files into wav files.

``` shell
ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s
```

You can do the conversion using ffmpeg  https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830). This operation might take several hours and should be only once.

3. Put all the wav files in a folder called `wav`. You should have something like `voxceleb2/wav/id*/*.wav` (e.g, `voxceleb2/wav/id00012/21Uxsk56VDQ/00001.wav`)
