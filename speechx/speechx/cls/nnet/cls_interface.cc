#include "cls/nnet/cls_nnet.h"
#include "cls/nnet/config.h"

namespace ppspeech{

void* cls_create_instance(const char* conf_path){
    Config conf(conf_path);
    //cls init
    ppspeech::ClsNnetConf cls_nnet_conf;
    cls_nnet_conf.wav_normal_ = conf.Read("wav_normal", true);
    cls_nnet_conf.wav_normal_type_ = conf.Read("wav_normal_type", std::string("linear"));
    cls_nnet_conf.wav_norm_mul_factor_ = conf.Read("wav_norm_mul_factor", 1.0);
    cls_nnet_conf.model_file_path_ = conf.Read("model_path", std::string(""));
    cls_nnet_conf.param_file_path_ = conf.Read("param_path", std::string(""));
    cls_nnet_conf.dict_file_path_ = conf.Read("dict_path", std::string(""));
    cls_nnet_conf.num_cpu_thread_ = conf.Read("num_cpu_thread", 12);
    cls_nnet_conf.samp_freq = conf.Read("samp_freq", 32000);
    cls_nnet_conf.frame_length_ms = conf.Read("frame_length_ms", 32);
    cls_nnet_conf.frame_shift_ms = conf.Read("frame_shift_ms", 10);
    cls_nnet_conf.num_bins = conf.Read("num_bins", 64);
    cls_nnet_conf.low_freq = conf.Read("low_freq", 50);
    cls_nnet_conf.high_freq = conf.Read("high_freq", 14000);
    cls_nnet_conf.dither = conf.Read("dither", 0.0);
    
    ppspeech::ClsNnet* cls_model = new ppspeech::ClsNnet();
    int ret = cls_model->init(cls_nnet_conf);
    return (void*)cls_model;
};

int cls_destroy_instance(void* instance){
    ppspeech::ClsNnet* cls_model = (ppspeech::ClsNnet*)instance;
    if(cls_model != NULL){
        delete cls_model;
        cls_model = NULL;
    }
    return 0;
};

int cls_feedforward(void* instance, const char* wav_path, int topk, char* result, int result_max_len){
    ppspeech::ClsNnet* cls_model = (ppspeech::ClsNnet*)instance;
    if(cls_model == NULL){
        printf("instance is null\n");
        return -1;
    }
    int ret = cls_model->forward(wav_path, topk, result, result_max_len);
    return 0;
};

int cls_reset(void* instance){
    ppspeech::ClsNnet* cls_model = (ppspeech::ClsNnet*)instance;
    if(cls_model == NULL){
        printf("instance is null\n");
        return -1;
    }
    cls_model->reset();
    return 0;
};

}