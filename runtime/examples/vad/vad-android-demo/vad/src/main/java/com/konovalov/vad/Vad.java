package com.konovalov.vad;

import android.text.Html;

import java.util.LinkedHashMap;
import java.util.LinkedList;

/**
 * Created by George Konovalov on 11/16/2019.
 */

public class Vad {
    public native boolean PPSVadCreateInstance();
    public native boolean PPSVadDestroyInstance();
    public native boolean PPSVadReset();
    public native int PPSVadChunkSizeSamples();
    public native boolean PPSVadFeedForward(short[] audio);

    private boolean needResetDetectedSamples = true;
    private long detectedVoiceSamplesMillis = 0;
    private long detectedSilenceSamplesMillis = 0;
    private long previousTimeMillis = System.currentTimeMillis();

    public Vad() {
    }

    public void start() {
        try {
            boolean result = PPSVadCreateInstance();

            if (result != true) {
                throw new RuntimeException("PPSVadCreateInstance error!");
            }
        } catch (Exception e) {
            throw new RuntimeException("Error can't start VAD!", e);
        }
    }

    public void stop() {
        try {
            boolean result = PPSVadDestroyInstance();
        } catch (Exception e) {
            throw new RuntimeException("PPSVadDestroyInstance error!", e);
        }
    }

    public boolean isSpeech(short[] audio) {
        if (audio == null) {
            throw new NullPointerException("Audio data is NULL!");
        }

        try {
            return PPSVadFeedForward(audio);
        } catch (Exception e) {
            throw new RuntimeException("PPSVadFeedForward error!", e);
        }
    }

    @Deprecated
    public void isContinuousSpeech(short[] audio, VadListener listener) {
        addContinuousSpeechListener(audio, listener);
    }

    public void addContinuousSpeechListener(short[] audio, VadListener listener) {
        if (audio == null) {
            throw new NullPointerException("Audio data is NULL!");
        }

        if (listener == null) {
            throw new NullPointerException("VadListener is NULL!");
        }

        long currentTimeMillis = System.currentTimeMillis();

        if (isSpeech(audio)) {
            needResetDetectedSamples = true;
            listener.onSpeechDetected();
        } else {
            if (needResetDetectedSamples) {
                needResetDetectedSamples = false;
            }
            listener.onNoiseDetected();
        }
    }

    static {
        System.loadLibrary("native-lib");
    }
}
