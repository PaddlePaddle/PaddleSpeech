// Write C++ code here.
//
// Do not forget to dynamically load the C++ library into your application.
//
// For instance,
//
// In MainActivity.java:
//    static {
//       System.loadLibrary("mysotest");
//    }
//
// Or, in MainActivity.kt:
//    companion object {
//      init {
//         System.loadLibrary("mysotest")
//      }
//    }

#include <jni.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "includes/vad_interface.h"
#include <android/log.h>
#include <fstream>

//如果你不想用引入头文件的方法，可以把导入头文件的include语句注释掉，然后将下面这句取消注释。
//string getStringFromSoLibrary();

void* vad_instance = nullptr;
int num_chunk = 0;
bool is_start = 0;
std::ofstream fp_pcm;
std::ofstream fp_log;

extern "C"
JNIEXPORT jboolean JNICALL Java_com_konovalov_vad_Vad_PPSVadCreateInstance(JNIEnv *env, jobject instance)
{
    if (vad_instance != nullptr) {
        int ret = PPSVadDestroyInstance(vad_instance);
        vad_instance = nullptr;
    }
    std::string conf_path = "/storage/self/primary/masimeng/vad/autotest/vad.conf";
    vad_instance = PPSVadCreateInstance(conf_path.c_str());
    __android_log_print(ANDROID_LOG_INFO,"[vad]", "vad instance create success");
    std::string fp_pcm_name = "/storage/self/primary/masimeng/vad/autotest/ori_pcm.pcm";
    std::string fp_log_name = "/storage/self/primary/masimeng/vad/autotest/ori_pcm.pcm.log";
    fp_pcm = std::ofstream(fp_pcm_name, std::ofstream::binary);
    fp_log = std::ofstream(fp_log_name);
    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_konovalov_vad_Vad_PPSVadDestroyInstance(JNIEnv *env, jobject instance){
    if (vad_instance != nullptr) {
        int ret =PPSVadDestroyInstance(vad_instance);
        vad_instance = nullptr;
        num_chunk = 0;
        fp_pcm.close();
        fp_log.close();
        __android_log_print(ANDROID_LOG_INFO,"[vad]", "vad instance destroy success");
    }
    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_konovalov_vad_Vad_PPSVadReset(JNIEnv *env, jobject instance){
    if (vad_instance != nullptr) {
        int ret = PPSVadReset(vad_instance);
        __android_log_print(ANDROID_LOG_INFO,"[vad]", "vad instance reset success");
    }
    return true;
}

extern "C"
JNIEXPORT jint JNICALL Java_com_konovalov_vad_Vad_PPSVadChunkSizeSamples(JNIEnv *env, jobject instance) {
    int ret = -1;
    if (vad_instance != nullptr) {
        ret = PPSVadChunkSizeSamples(vad_instance);
    }
    return ret;
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_konovalov_vad_Vad_PPSVadFeedForward(JNIEnv *env, jobject instance, jshortArray audio){
    jshort *arrayElements = (*env).GetShortArrayElements(audio, 0);
    jint array_size = (*env).GetArrayLength(audio);
    fp_pcm.write(reinterpret_cast<const char *>(arrayElements), array_size * sizeof(jshort));

    if (vad_instance != nullptr) {
        int chunk_size = PPSVadChunkSizeSamples(vad_instance);
//        __android_log_print(ANDROID_LOG_INFO,"[vad]", "chunk size is : %d", chunk_size);
        if (array_size != chunk_size){
            __android_log_print(ANDROID_LOG_INFO,"[vad]", "error array_size : %d != chunk_size : %d", array_size, chunk_size);
            return false;
        }
        std::vector<float> chunk(array_size, 0);
//        __android_log_print(ANDROID_LOG_INFO,"[vad]", "array size : %d array[0]: %d", array_size, arrayElements[0]);

        for (int i = 0; i < array_size; ++i){
            chunk[i] = (float)arrayElements[i];
        }
        PPSVadState_t ret = PPSVadFeedForward(vad_instance, chunk.data(), array_size);
        if (ret == PPS_VAD_START){
            __android_log_print(ANDROID_LOG_INFO,"[vad]", "st:%s s", std::to_string(num_chunk * chunk_size * 1.0 / 16 / 1000).c_str());
            is_start = true;
            fp_log << "st: " << std::to_string(num_chunk * chunk_size * 1.0 / 16 / 1000).c_str() << "s ";
        } else if(is_start == true && ret == PPS_VAD_END){
            __android_log_print(ANDROID_LOG_INFO,"[vad]", "et:%s s", std::to_string(num_chunk * chunk_size * 1.0 / 16 / 1000).c_str());
            is_start = false;
            fp_log << "et: " << std::to_string(num_chunk * chunk_size * 1.0 / 16 / 1000).c_str() << "s\n";
        }
        num_chunk ++;
//        __android_log_print(ANDROID_LOG_INFO,"[vad]", "num_chunk:%d", num_chunk);
        if (ret == PPS_VAD_START || ret == PPS_VAD_SPEECH){
            return true;
        }
        return false;
    }
    return false;
}