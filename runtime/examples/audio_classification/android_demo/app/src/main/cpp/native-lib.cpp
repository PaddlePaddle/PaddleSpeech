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
#include "includes/panns_interface.h"

//如果你不想用引入头文件的方法，可以把导入头文件的include语句注释掉，然后将下面这句取消注释。
//string getStringFromSoLibrary();

void* cls_instance = nullptr;

extern "C"
JNIEXPORT jboolean JNICALL Java_com_baidu_paddlespeech_cls_MainActivity_nClsCreateInstance(JNIEnv *env, jobject instance)
{
    if (cls_instance != nullptr) {
        ppspeech::ClsDestroyInstance(cls_instance);
        cls_instance = nullptr;
    }
    std::string conf_path = "/data/local/tmp/masimeng/cls/conf";
    cls_instance = ppspeech::ClsCreateInstance(conf_path.c_str());
    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_baidu_paddlespeech_cls_MainActivity_nClsDestroyInstance(JNIEnv *env, jobject instance){
    if (cls_instance != nullptr) {
        ppspeech::ClsDestroyInstance(cls_instance);
        cls_instance = nullptr;
    }
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL Java_com_baidu_paddlespeech_cls_MainActivity_nClsFeedForward(JNIEnv *env, jobject instance){
    if (cls_instance != nullptr) {
        char result[1024] = {0};
        std::string wav_path = "/data/local/tmp/masimeng/cls/test.wav";
        int ret = ppspeech::ClsFeedForward(cls_instance, wav_path.c_str(), 1, result, 1024);
        return env->NewStringUTF(result);
    }
    return env->NewStringUTF(NULL);
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_baidu_paddlespeech_cls_MainActivity_nClsReset(JNIEnv *env, jobject instance){
    if (cls_instance != nullptr) {
        ppspeech::ClsReset(cls_instance);
    }
    return true;
}