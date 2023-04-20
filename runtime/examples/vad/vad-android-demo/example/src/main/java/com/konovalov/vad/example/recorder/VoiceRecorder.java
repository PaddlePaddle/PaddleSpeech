package com.konovalov.vad.example.recorder;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import com.konovalov.vad.example.recorder.VoiceRecorderConfig;

import com.konovalov.vad.Vad;
//import com.konovalov.vad.VadConfig;
import com.konovalov.vad.VadListener;

import static android.media.AudioFormat.CHANNEL_IN_MONO;
import static android.media.AudioFormat.CHANNEL_IN_STEREO;

import java.io.File;

/**
 * Created by George Konovalov on 11/16/2019.
 */

public class VoiceRecorder {
    private static final int PCM_CHANNEL = CHANNEL_IN_MONO;
    private static final int PCM_ENCODING_BIT = AudioFormat.ENCODING_PCM_16BIT;

    private VoiceRecorderConfig config;
    private Vad vad;
    private AudioRecord audioRecord;
    private Listener callback;
    private Thread thread;

    private boolean isListening = false;

    private static final String TAG = VoiceRecorder.class.getSimpleName();

    public VoiceRecorder(Listener callback, VoiceRecorderConfig config) {
        this.callback = callback;
        this.config = config;
        this.vad = new Vad();
    }

    public void updateConfig(VoiceRecorderConfig config) {
        this.config = config;
    }

    public void start() {
        stop();
        audioRecord = createAudioRecord();
        if (audioRecord != null) {
            isListening = true;
            audioRecord.startRecording();

            thread = new Thread(new ProcessVoice());
            thread.start();
            vad.start();
        } else {
            Log.w(TAG, "Failed start Voice Recorder!");
        }
    }


    public void stop() {
        isListening = false;
        if (thread != null) {
            thread.interrupt();
            thread = null;
        }
        if (audioRecord != null) {
            try {
                audioRecord.release();
            } catch (Exception e) {
                Log.e(TAG, "Error stop AudioRecord ", e);
            }
            audioRecord = null;
        }
        if (vad != null) {
            vad.stop();
        }
    }


    private AudioRecord createAudioRecord() {
        try {
            final int minBufSize = AudioRecord.getMinBufferSize(config.getSampleRate().getValue(), PCM_CHANNEL, PCM_ENCODING_BIT);
            int frame_size = config.getFrameSize().getValue();
            if (minBufSize > frame_size) {
                Log.e(TAG, "minBufSize > frame_size");
                return null;
            }
            Log.i(TAG, "minBufSize : " + minBufSize);
            final AudioRecord audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, config.getSampleRate().getValue(), PCM_CHANNEL, PCM_ENCODING_BIT, frame_size);
            Log.i(TAG, "config.getSampleRate().getValue() : " + config.getSampleRate().getValue());

            if (audioRecord.getState() == AudioRecord.STATE_INITIALIZED) {
                return audioRecord;
            } else {
                audioRecord.release();
            }
        } catch (IllegalArgumentException e) {
            Log.e(TAG, "Error can't create AudioRecord ", e);
        }

        return null;
    }

    private int getNumberOfChannels() {
        switch (PCM_CHANNEL) {
            case CHANNEL_IN_MONO:
                return 1;
            case CHANNEL_IN_STEREO:
                return 2;
        }
        return 1;
    }

    private class ProcessVoice implements Runnable {

        @Override
        public void run() {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);
            final int minBufSize = AudioRecord.getMinBufferSize(config.getSampleRate().getValue(), PCM_CHANNEL, PCM_ENCODING_BIT);
            while (!Thread.interrupted() && isListening && audioRecord != null) {
                short[] buffer = new short[config.getFrameSize().getValue()];
                audioRecord.read(buffer, 0, buffer.length);
                detectSpeech(buffer);
            }
        }

        private void detectSpeech(short[] buffer) {
            vad.addContinuousSpeechListener(buffer, new VadListener() {
                @Override
                public void onSpeechDetected() {
                    callback.onSpeechDetected();
                }

                @Override
                public void onNoiseDetected() {
                    callback.onNoiseDetected();
                }
            });
        }
    }

    public interface Listener {
        void onSpeechDetected();

        void onNoiseDetected();
    }

}
