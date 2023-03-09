// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// PackageName: paddlespeech.baidu.com
// ClassName: vadjni
#include <jni.h>

#include "vad/interface/vad_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef jlong PPSJniHandle_t;

JNIEXPORT PPSJniHandle_t JNICALL
Java_com_baidu_paddlespeech_vadjni_createInstance(JNIEnv* env,
                                                  jobject thiz,
                                                  jstring conf_path);

JNIEXPORT jint JNICALL Java_com_baidu_paddlespeech_vadjni_destoryInstance(
    JNIEnv* env, jobject thiz, PPSJniHandle_t instance);


JNIEXPORT jint JNICALL Java_com_baidu_paddlespeech_vadjni_reset(
    JNIEnv* env, jobject thiz, PPSJniHandle_t instance);

JNIEXPORT jint JNICALL Java_com_baidu_paddlespeech_vadjni_chunkSizeSamples(
    JNIEnv* env, jobject thiz, PPSJniHandle_t instance);

// typedef enum {
//     PPS_VAD_ILLEGAL = 0,  // error
//     PPS_VAD_SIL,          // silence
//     PPS_VAD_START,        // start speech
//     PPS_VAD_SPEECH,       // in speech
//     PPS_VAD_END,          // end speech
//     PPS_VAD_NUMSTATES,    // number of states
// } PPSVadState_t;

JNIEXPORT jint JNICALL Java_com_baidu_paddlespeech_vadjni_feedForward(
    JNIEnv* env, jobject thiz, PPSJniHandle_t instance, jfloatArray chunk);

#ifdef __cplusplus
}
#endif