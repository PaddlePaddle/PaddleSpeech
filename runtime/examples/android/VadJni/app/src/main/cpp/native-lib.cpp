
#include <string>
#include "vad_interface.h"
#include <jni.h>

extern "C"
JNIEXPORT jstring JNICALL
Java_com_baidu_paddlespeech_vadjni_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_baidu_paddlespeech_vadjni_MainActivity_createInstance(
        JNIEnv* env,
        jobject thiz,
        jstring conf_path){
    const char* path = env->GetStringUTFChars(conf_path, JNI_FALSE);
    PPSHandle_t handle = PPSVadCreateInstance(path);

    return (jlong)(handle);
    return 0;
}


extern "C"
JNIEXPORT jint JNICALL
Java_com_baidu_paddlespeech_vadjni_MainActivity_destroyInstance(JNIEnv *env, jobject thiz,
                                                                jlong instance) {
    PPSHandle_t handle = (PPSHandle_t)(instance);
    return (jint)PPSVadDestroyInstance(handle);
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_baidu_paddlespeech_vadjni_MainActivity_reset(JNIEnv *env, jobject thiz, jlong instance) {
    PPSHandle_t handle = (PPSHandle_t)(instance);
    return (jint)PPSVadReset(handle);
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_baidu_paddlespeech_vadjni_MainActivity_chunkSizeSamples(JNIEnv *env, jobject thiz,
                                                                 jlong instance) {
    PPSHandle_t handle = (PPSHandle_t)(instance);
    return (jint)PPSVadChunkSizeSamples(handle);
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_baidu_paddlespeech_vadjni_MainActivity_feedForward(JNIEnv *env, jobject thiz,
                                                            jlong instance, jfloatArray chunk) {
    PPSHandle_t handle = (PPSHandle_t)(instance);
    jsize num_elms = env->GetArrayLength(chunk);
    jfloat* chunk_ptr = env->GetFloatArrayElements(chunk, JNI_FALSE);
    return (jint)PPSVadFeedForward(handle, (float*)chunk_ptr, (int)num_elms);
}