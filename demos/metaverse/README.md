# Metaverse

## Introduction
Metaverse is a new Internet application and social form integrating virtual reality produced by integrating a variety of new technologies. 

This demo is an implementation to let a celebrity in an image "speak". With the composition of `TTS` mudule of `PaddleSpeech` and `PaddleGAN`, we integrate the installation and the specific modules in a single shell script. 

## Usage

You can make your favorite person say the specified content with the `TTS` mudule of `PaddleSpeech` and `PaddleGAN`, and construct your own virtual human.

Run `run.sh` to complete all the essential procedures, including the installation.  

```bash
./run.sh
```
In `run.sh`, it will execute `source path.sh` firstly, which will set the environment variants. 

If you would like to try your own image, please replace the image `download/Lamarr.png` in the shell script.

The result has shown on our [notebook](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/tutorial/tts/tts_tutorial.ipynb).
