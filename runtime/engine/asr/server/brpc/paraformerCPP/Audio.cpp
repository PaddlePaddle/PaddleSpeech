#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <webrtc_vad.h>
#include "ComDefine.h"
#include "Audio.h"

using namespace std;

class AudioWindow {
  private:
    int *window;
    int in_idx;
    int out_idx;
    int sum;
    int window_size = 0;

  public:
    AudioWindow(int window_size) : window_size(window_size)
    {
        window = (int *)calloc(sizeof(int), window_size + 1);
        in_idx = 0;
        out_idx = 1;
        sum = 0;
    };
    ~AudioWindow(){
        free(window);
    };
    int put(int val)
    {
        sum = sum + val - window[out_idx];
        window[in_idx] = val;
        in_idx = in_idx == window_size ? 0 : in_idx + 1;
        out_idx = out_idx == window_size ? 0 : out_idx + 1;
        return sum;
    };
};

AudioFrame::AudioFrame(){};
AudioFrame::AudioFrame(int len) : len(len)
{
    start = 0;
};
AudioFrame::~AudioFrame(){};
int AudioFrame::set_start(int val)
{
    start = val < 0 ? 0 : val;
    return start;
};

int AudioFrame::set_end(int val, int max_len)
{

    float num_samples = val - start;
    float frame_length = 400;
    float frame_shift = 160;
    float num_new_samples =
        ceil((num_samples - frame_length) / frame_shift) * frame_shift + frame_length;

    end = start + num_new_samples;
    len = (int)num_new_samples;
    if (end > max_len){
        printf("frame end > max_len!!!!!!!\n");
    }
        
    return end;
};

int AudioFrame::get_start()
{
    return start;
};

int AudioFrame::get_len()
{
    return len;
};

int AudioFrame::disp()
{
    printf("not imp!!!!\n");

    return 0;
};

Audio::Audio(int data_type) : data_type(data_type)
{
    speech_buff = NULL;
    align_size = 1360;
}

Audio::Audio(int data_type, int size) : data_type(data_type)
{
    speech_buff = NULL;
    align_size = (float)size;
}

Audio::~Audio()
{
    if (speech_buff != NULL) {
        free(speech_buff);
        speech_data.clear();
    }
}

void Audio::disp()
{
    printf("Audio time is %f s. len is %d\n", (float)speech_len / 16000,
           speech_len);
}

void Audio::loadwavfrommem(AudioFile<float>audio)
{
    if (speech_buff != NULL) {
        free(speech_buff);
        speech_data.clear();
    }
    int wav_length = audio.getNumSamplesPerChannel();
    int channelNum = audio.getNumChannels();

    speech_len = wav_length * channelNum;
    printf("wav_length:%d, channelNum: %d", wav_length, channelNum);
    
    speech_align_len = (int)(ceil((float)speech_len / align_size) * align_size);
    speech_buff = (int16_t *)malloc(sizeof(int16_t) * speech_align_len);
    memset(speech_buff, 0, sizeof(int16_t) * speech_align_len);
   
    for (int i = 0; i < wav_length; i++)
    {
        for (int channel = 0; channel < channelNum; channel++)
        {
            speech_buff[i * channelNum + channel] = (int16_t)(audio.samples[channel][i] * 32768);
        }
    }
    
    for (int i = 0; i < speech_len; i++) {
        float temp = (float)speech_buff[i];
        speech_data.emplace_back(temp);
    }
    
    AudioFrame *frame = new AudioFrame(speech_len);
    frame_queue.push(frame);
}

int Audio::fetch(vector<float> &dout, int &len, int &flag)
{
    if (frame_queue.size() > 0) {
        AudioFrame *frame = frame_queue.front();
        frame_queue.pop();
        len = frame->get_len();
        int speech_len = speech_data.size();
        auto last = min(speech_len, frame->get_start() + len);
        dout.insert(dout.begin(), speech_data.begin() + frame->get_start(), speech_data.begin() + last);
        delete frame;
        flag = S_END;
        return 1;
    } else {
        return 0;
    }
}


#define UNTRIGGERED 0
#define TRIGGERED   1

#define SPEECH_LEN_5S  (16000 * 5)
#define SPEECH_LEN_10S (16000 * 10)
#define SPEECH_LEN_15S (16000 * 15)
#define SPEECH_LEN_20S (16000 * 20)
#define SPEECH_LEN_30S (16000 * 30)
#define SPEECH_LEN_60S (16000 * 60)

void Audio::split()
{
    VadInst *handle = WebRtcVad_Create();
    WebRtcVad_Init(handle);
    WebRtcVad_set_mode(handle, 2);
    int window_size = 10;
    AudioWindow audiowindow(window_size);
    int status = UNTRIGGERED;
    int offset = 0;
    int fs = 16000;
    int step = 160;

    AudioFrame *frame;

    frame = frame_queue.front();
    frame_queue.pop();
    delete frame;

    while (offset < speech_len - step) {
        int n = WebRtcVad_Process(handle, fs, speech_buff + offset, step);
        
        if (status == UNTRIGGERED && audiowindow.put(n) >= window_size - 1) {
            frame = new AudioFrame();
            int start = offset - step * (window_size - 1);
            frame->set_start(start);
            status = TRIGGERED;
        } else if (status == TRIGGERED) {
            int win_weight = audiowindow.put(n);
            int voice_len = (offset - frame->get_start());
            int gap = 0;
            if (voice_len < SPEECH_LEN_5S) {
                offset += step;
                continue;
            } else if (voice_len < SPEECH_LEN_10S) {
                gap = 1;
            } else if (voice_len < SPEECH_LEN_20S) {
                gap = window_size / 5;
            } else {
                gap = window_size - 1;
            }

            if (win_weight < gap || voice_len >= SPEECH_LEN_15S) {
                status = UNTRIGGERED;
                offset = frame->set_end(offset, speech_align_len);
                frame_queue.push(frame);
                frame = NULL;
            }
        }
        offset += step;
    }

    if (frame != NULL) {
        frame->set_end(speech_len, speech_align_len);
        frame_queue.push(frame);
        frame = NULL;
    }
    WebRtcVad_Free(handle);
}
