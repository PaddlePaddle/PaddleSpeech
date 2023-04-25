This VAD library can process audio in real-time utilizing 
[Gaussian Mixture Model](http://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model) (GMM)
which helps identify presence of human speech in an audio sample that contains a mixture of speech 
and noise. VAD work offline and all processing done on device.
  
Library based on 
[WebRTC VAD](https://chromium.googlesource.com/external/webrtc/+/branch-heads/43/webrtc/common_audio/vad/) 
from Google which is reportedly one of the best available: it's fast, modern and free.
This  algorithm has  found  wide adoption and has recently become one of 
the gold-standards for delay-sensitive scenarios like web-based interaction.
  
If you are looking for a higher accuracy and faster processing time I recommend to use Deep Neural 
Networks(DNN). Please see for reference the following paper with 
[DNN vs GMM](https://www.microsoft.com/en-us/research/uploads/prod/2018/02/KoPhiliposeTashevZarar_ICASSP_2018.pdf)
comparison.

<p align="center">
<img src="https://raw.githubusercontent.com/gkonovalov/android-vad/master/demo.gif" alt="drawing" height="400"/>
</p>

## Parameters
VAD library only accepts 16-bit mono PCM audio stream and can work with next Sample Rates, Frame Sizes and Classifiers. 
<table>
<tr>
<td>
&nbsp

| Valid Sample Rate  | Valid Frame Size  |   
|:-------------------|:------------------|   
| 8000Hz             | 80, 160, 240      |  
| 16000Hz            | 160, 320, 480     |   
| 32000Hz            | 320, 640, 960     |   
| 48000Hz            | 480, 960, 1440    |   
</td>
<td>
&nbsp

| Valid Classifiers |
|:------------------|
| NORMAL            |
| LOW_BITRATE       |
| AGGRESSIVE        |
| VERY_AGGRESSIVE   |
</td>
</tr>
</table>


**Silence duration (ms)** - this parameter used in Continuous Speech detector,
the value of this parameter will define the necessary and sufficient 
duration of negative results to recognize it as silence.
 
**Voice duration (ms)** - this parameter used in Continuous Speech detector,
the value of this parameter will define the necessary and sufficient 
duration of positive results to recognize result as speech.

Recommended parameters:
* Sample Rate - **16KHz**,
* Frame Size - **160**,
* Mode - **VERY_AGGRESSIVE**,
* Silence Duration - **500ms**,
* Voice Duration - **500ms**;

## Usage
VAD supports 2 different ways of detecting speech:
1. Continuous Speech listener was designed to detect long utterances 
without returning false positive results when user makes pauses between 
sentences.
```java
 Vad vad = new Vad(VadConfig.newBuilder()
                .setSampleRate(VadConfig.SampleRate.SAMPLE_RATE_16K)
                .setFrameSize(VadConfig.FrameSize.FRAME_SIZE_160)
                .setMode(VadConfig.Mode.VERY_AGGRESSIVE)
                .setSilenceDurationMillis(500)
                .setVoiceDurationMillis(500)
                .build());

        vad.start();
        
        vad.addContinuousSpeechListener(short[] audioFrame, new VadListener() {
            @Override
            public void onSpeechDetected() {
                //speech detected!
            }

            @Override
            public void onNoiseDetected() {
                //noise detected!
            }
        });
        
        vad.stop();
```

2. Speech detector was designed to detect speech/noise in small audio 
frames and return result for every frame. This method will not work for 
long utterances.
```java
 Vad vad = new Vad(VadConfig.newBuilder()
                .setSampleRate(VadConfig.SampleRate.SAMPLE_RATE_16K)
                .setFrameSize(VadConfig.FrameSize.FRAME_SIZE_160)
                .setMode(VadConfig.Mode.VERY_AGGRESSIVE)
                .build());

        vad.start();
        
        boolean isSpeech = vad.isSpeech(short[] audioFrame);
        
        vad.stop();
```
## Requirements
Android VAD supports Android 4.1 (Jelly Bean) and later.

## Development

To open the project in Android Studio:

1. Go to *File* menu or the *Welcome Screen*
2. Click on *Open...*
3. Navigate to VAD's root directory.
4. Select `setting.gradle`

## Download
[![](https://jitpack.io/v/gkonovalov/android-vad.svg)](https://jitpack.io/#gkonovalov/android-vad)


Gradle is the only supported build configuration, so just add the dependency to your project `build.gradle` file:
1. Add it in your root build.gradle at the end of repositories:
```groovy
allprojects {
   repositories {
     maven { url 'https://jitpack.io' }
   }
}
```

2. Add the dependency
```groovy
dependencies {
    implementation 'com.github.gkonovalov:android-vad:1.0.1'
}
```

You also can download precompiled AAR library and APK files from GitHub's [releases page](https://github.com/gkonovalov/android-vad/releases).

------------
Georgiy Konovalov 2021 (c) [MIT License](https://opensource.org/licenses/MIT)