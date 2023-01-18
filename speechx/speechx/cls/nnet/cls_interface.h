#pragma once

namespace ppspeech{

void* cls_create_instance(const char* conf_path);
int cls_destroy_instance(void* instance);
int cls_feedforward(void* instance, const char* wav_path, int topk, char* result, int result_max_len);
int cls_reset(void* instance);

}