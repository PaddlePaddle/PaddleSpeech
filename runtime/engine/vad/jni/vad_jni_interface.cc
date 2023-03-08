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

#include "vad/jni/vad_jni_interface.h"

JNIEXPORT jlong JNICALL Java_com_baidu_paddlespeech_PPSVadJni_createInstance(
    JNIEnv* env, jobject thiz, jstring conf_path) {
    const char* path = env->GetStringUTFChars(conf_path, JNI_FALSE);
    PPSHandle_t handle = PPSVadCreateInstance(path);

    return (jlong)(handle);
}

JNIEXPORT jint JNICALL Java_com_baidu_paddlespeech_PPSVadJni_destoryInstance(
    JNIEnv* env, jobject thiz, PPSJniHandle_t instance) {
    PPSHandle_t handle = (PPSHandle_t)(instance);
    return (jint)PPSVadDestroyInstance(handle);
}


JNIEXPORT jint JNICALL Java_com_baidu_paddlespeech_PPSVadJni_reset(
    JNIEnv* env, jobject thiz, PPSJniHandle_t instance) {
    PPSHandle_t handle = (PPSHandle_t)(instance);
    return (jint)PPSVadReset(handle);
}

JNIEXPORT jint JNICALL Java_com_baidu_paddlespeech_PPSVadJni_chunkSizeSamples(
    JNIEnv* env, jobject thiz, PPSJniHandle_t instance) {
    PPSHandle_t handle = (PPSHandle_t)(instance);
    return (jint)PPSVadChunkSizeSamples(handle);
}

JNIEXPORT jint JNICALL Java_com_baidu_paddlespeech_PPSVadJni_feedForward(
    JNIEnv* env, jobject thiz, PPSJniHandle_t instance, jfloatArray chunk) {
    PPSHandle_t handle = (PPSHandle_t)(instance);
    jsize num_elms = env->GetArrayLength(chunk);
    jfloat* chunk_ptr = env->GetFloatArrayElements(chunk, JNI_FALSE);
    return (jint)PPSVadFeedForward(handle, (float*)chunk_ptr, (int)num_elms);
}