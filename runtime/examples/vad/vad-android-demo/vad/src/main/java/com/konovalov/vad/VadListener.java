package com.konovalov.vad;

/**
 * Created by George Konovalov on 11/16/2019.
 */


public interface VadListener {
    void onSpeechDetected();

    void onNoiseDetected();
}
