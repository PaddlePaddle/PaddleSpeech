package com.konovalov.vad.example.recorder;


/**
 * Created by George Konovalov on 11/16/2019.
 */

public class VoiceRecorderConfig {
    private SampleRate sampleRate;
    private Mode mode;
    private FrameSize frameSize;
    private int voiceDurationMillis;
    private int silenceDurationMillis;

    public VoiceRecorderConfig() {
    }

    public VoiceRecorderConfig(Builder builder) {
        this.voiceDurationMillis = builder.voiceDurationMillis;
        this.silenceDurationMillis = builder.silenceDurationMillis;
        this.sampleRate = builder.sampleRate;
        this.frameSize = builder.frameSize;
        this.mode = builder.mode;
    }

    public SampleRate getSampleRate() {
        return sampleRate;
    }

    public Mode getMode() {
        return mode;
    }

    public FrameSize getFrameSize() {
        return frameSize;
    }

    public int getVoiceDurationMillis() {
        return voiceDurationMillis;
    }

    public int getSilenceDurationMillis() {
        return silenceDurationMillis;
    }

    public void setSampleRate(SampleRate sampleRate) {
        this.sampleRate = sampleRate;
    }

    public void setMode(Mode mode) {
        this.mode = mode;
    }

    public void setFrameSize(FrameSize frameSize) {
        this.frameSize = frameSize;
    }

    public void setVoiceDurationMillis(int voiceDurationMillis) {
        this.voiceDurationMillis = voiceDurationMillis;
    }

    public void setSilenceDurationMillis(int silenceDurationMillis) {
        this.silenceDurationMillis = silenceDurationMillis;
    }

    public static Builder newBuilder() {
        return new Builder();
    }

    public static class Builder {
        private SampleRate sampleRate = SampleRate.SAMPLE_RATE_16K;
        private Mode mode = Mode.VERY_AGGRESSIVE;
        private FrameSize frameSize;
        private int voiceDurationMillis = 500;
        private int silenceDurationMillis = 500;

        private Builder() {
        }

        public Builder setSampleRate(SampleRate sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }

        public Builder setMode(Mode mode) {
            this.mode = mode;
            return this;
        }

        public Builder setFrameSize(FrameSize frameSize) {
            this.frameSize = frameSize;
            return this;
        }

        public Builder setVoiceDurationMillis(int voiceDurationMillis) {
            this.voiceDurationMillis = voiceDurationMillis;
            return this;
        }

        public Builder setSilenceDurationMillis(int silenceDurationMillis) {
            this.silenceDurationMillis = silenceDurationMillis;
            return this;
        }

        public VoiceRecorderConfig build() {
            return new VoiceRecorderConfig(this);
        }
    }

    public enum SampleRate {
        SAMPLE_RATE_16K(16000);

        private int sampleRate;

        public int getValue() {
            return sampleRate;
        }

        SampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
        }
    }

    public enum Mode {
        NORMAL(0),
        LOW_BITRATE(1),
        AGGRESSIVE(2),
        VERY_AGGRESSIVE(3);

        private int mode;

        public int getValue() {
            return mode;
        }

        Mode(int mode) {
            this.mode = mode;
        }
    }

    public enum FrameSize {
        FRAME_SIZE_1536(1536);

        private int frameSize;

        public int getValue() {
            return frameSize;
        }

        FrameSize(int frameSize) {
            this.frameSize = frameSize;
        }
    }



}
