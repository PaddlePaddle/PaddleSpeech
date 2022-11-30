# 语音合成 Java API Demo 使用指南

在 Android 上实现语音合成功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。

本文主要介绍语音合成 Demo 运行方法。

## 如何运行语音合成 Demo

### 环境准备

1. 在本地环境安装好 Android Studio 工具，详细安装方法请见 [Android Stuido 官网](https://developer.android.com/studio)。
2. 准备一部 Android 手机，并开启 USB 调试模式。开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`。

**注意**：
> 如果您的 Android Studio 尚未配置 NDK ，请根据 Android Studio 用户指南中的[安装及配置 NDK 和 CMake ](https://developer.android.com/studio/projects/install-ndk)内容，预先配置好 NDK 。您可以选择最新的 NDK 版本，或者使用 Paddle Lite 预测库版本一样的 NDK。

### 部署步骤

1. 用 Android Studio 打开 TTSAndroid 工程。
2. 手机连接电脑，打开 USB 调试和文件传输模式，并在 Android Studio 上连接自己的手机设备（手机需要开启允许从 USB 安装软件权限）。

**注意：**
>1. 如果您在导入项目、编译或者运行过程中遇到 NDK 配置错误的提示，请打开 `File > Project Structure > SDK Location`，修改 `Andriod NDK location` 为您本机配置的 NDK 所在路径。
>2. 如果您是通过 Andriod Studio 的 SDK Tools 下载的 NDK (见本章节"环境准备")，可以直接点击下拉框选择默认路径。
>3. 还有一种 NDK 配置方法，你可以在 `TTSAndroid/local.properties` 文件中手动添加 NDK 路径配置 `nkd.dir=/root/android-ndk-r20b`
>4. 如果以上步骤仍旧无法解决 NDK 配置错误，请尝试根据 Andriod Studio 官方文档中的[更新 Android Gradle 插件](https://developer.android.com/studio/releases/gradle-plugin?hl=zh-cn#updating-plugin)章节，尝试更新 Android Gradle plugin 版本。

3. 点击 Run 按钮，自动编译 APP 并安装到手机。(该过程会自动下载 Paddle Lite 预测库和模型，需要联网)
   成功后效果如下：
    - pic 1：APP 安装到手机。
    - pic 2：APP 打开后的效果，在下拉框中选择待合成的文本。
    - pic 3：合成后点击按钮播放音频。

<p align="center"><img width="350" height="500"  src="https://user-images.githubusercontent.com/24568452/204450217-d166588a-5341-4565-8662-0f8129284bba.png"/><img width="350" height="500" src="https://user-images.githubusercontent.com/24568452/204450231-d6f3105c-276a-4af5-a3ba-864d9f5ee24e.png"/><img width="350" height="500" src="https://user-images.githubusercontent.com/24568452/204450269-0ddf46ec-eedd-4c90-8a0d-e915622fdf3e.png"/></p>

## 更新预测库

* Paddle Lite
  项目：[https://github.com/PaddlePaddle/Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)。


参考 [Paddle Lite 源码编译文档](https://www.paddlepaddle.org.cn/lite/v2.11/source_compile/compile_env.html)，编译
Android 预测库。

* 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
* 替换 java 库
    * jar 包
      将生成的 `build.lite.android.xxx.gcc/inference_lite_lib.android.xxx/java/jar/PaddlePredictor.jar`
      替换 Demo 中的 `TTSAndroid/app/libs/PaddlePredictor.jar`。
    * Java so
        * arm64-v8a
          将生成的 `build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/java/so/libpaddle_lite_jni.so`
          库替换 Demo 中的 `TTSAndroid/app/src/main/jniLibs/arm64-v8a/libpaddle_lite_jni.so`。

## Demo 内容介绍

先整体介绍下目标检测 Demo 的代码结构，然后介绍 Java 各功能模块的功能。

<p align="center">
<img width="442" alt="image" src="https://user-images.githubusercontent.com/24568452/204455080-4f96fe55-6058-4235-bb92-cc98cfcc8bb6.png">
</p>

### 重点关注内容

1. `Predictor.java`： 预测代码。

```bash
# 位置：
TTSAndroid/app/src/main/java/com/baidu/paddle/lite/demo/tts/Predictor.java
```

2. `fastspeech2_csmsc_arm.nb`  和 `mb_melgan_csmsc_arm.nb`: 模型文件 (opt 工具转化后 Paddle Lite 模型)
   ，分别来自 [fastspeech2_cnndecoder_csmsc_pdlite_1.3.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_pdlite_1.3.0.zip)
   和 [mb_melgan_csmsc_pdlite_1.3.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_pdlite_1.3.0.zip)。

```bash
# 位置：
TTSAndroid/app/src/main/assets/models/cpu/fastspeech2_csmsc_arm.nb
TTSAndroid/app/src/main/assets/models/cpu/mb_melgan_csmsc_arm.nb
```

3. `libpaddle_lite_jni.so`、`PaddlePredictor.jar`：Paddle Lite Java 预测库与 jar 包。

```bash
# 位置
TTSAndroid/app/src/main/jniLibs/arm64-v8a/libpaddle_lite_jni.so
TTSAndroid/app/libs/PaddlePredictor.jar
```

> 如果要替换动态库 so 和 jar 文件，则将新的动态库 so 更新到 `TTSAndroid/app/src/main/jniLibs/arm64-v8a/` 目录下 新的 jar 文件更新到 `TTSAndroid/app/libs/` 目录下

4. `build.gradle` : 定义编译过程的 gradle 脚本。（不用改动，定义了自动下载 Paddle Lite 预测和模型的过程）

```bash
# 位置
TTSAndroid/app/build.gradle
```

如果需要手动更新模型和预测库，则可将 gradle 脚本中的 `download*` 接口注释即可, 将新的预测库替换至相应目录下

### Java 端

* 模型存放，将下载好的模型解压存放在 `app/src/assets/models` 目录下。
* TTSAndroid Java 包在 `app/src/main/java/com/baidu/paddle/lite/demo/tts` 目录下，实现 APP 界面消息事件。
* MainActivity 实现 APP 的创建、运行、释放功能，重点关注 `onLoadModel` 和 `onRunModel` 函数，实现 APP 界面值传递和推理处理。

     ```java
    public boolean onLoadModel() {
        return predictor.init(MainActivity.this, modelPath, AMmodelName, VOCmodelName, cpuThreadNum,
                cpuPowerMode);
    }
     
    public boolean onRunModel() {
        return predictor.isLoaded() && predictor.runModel(phones);
    }
     ```

* SettingActivity 实现设置界面各个元素的更新与显示如模型地址、线程数、输入 shape 大小等，如果新增/删除界面的某个元素，均在这个类里面实现：
    - 参数的默认值可在 `app/src/main/res/values/strings.xml` 查看
    - 每个元素的 ID 和 value 是对应 `app/src/main/res/xml/settings.xml`
      和 `app/src/main/res/values/string.xml` 文件中的值
    - 这部分内容不建议修改，如果有新增属性，可以按照此格式进行添加

* Predictor 使用 Java API 实现语音合成模型的预测功能，重点关注 `init`、和 `runModel` 函数，实现 Paddle Lite 端侧推理功能：
     ```java
     // 初始化函数，完成预测器初始化
     public boolean init(Context appCtx, String modelPath, String AMmodelName, String VOCmodelName, int cpuThreadNum, String cpuPowerMode);
     // 模型推理函数
     public boolean runModel(float[] phones);
     ```

## 代码讲解 （使用 Paddle Lite `Java API` 执行预测）

Android 示例基于 Java API 开发，调用 Paddle Lite `Java API` 包括以下五步。更详细的 `API`
描述参考：[Paddle Lite Java API ](https://www.paddlepaddle.org.cn/lite/v2.11/api_reference/java_api_doc.html)。

## 如何更新模型和输入

### 更新模型

1. 将优化后的模型存放到目录 `TTSAndroid/app/src/main/assets/models/cpu/`
   下，可任意换成 [released_model.md](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md)
   中的 `*_pdlite_*.zip/*_arm.nb`
   格式的声学模型和声码器，注意更换声学模型需要对应修改 `TTSAndroid/app/src/main/java/com/baidu/paddle/lite/demo/tts/MainActivity.java`
   中的 `sentencesToChoose` 数组。
2. 如果模型名字跟工程中模型名字一模一样，即均是使用`fastspeech2_csmsc_arm.nb` （假设声学模型的 `phone_id_map.txt`
   也一样）和 `mb_melgan_csmsc_arm.nb`
   ，则代码不需更新；否则，需要修改  `TTSAndroid/app/src/main/java/com/baidu/paddle/lite/demo/tts/MainActivity.java`
   中的 `AMmodelName` 和 `VOCmodelName`：

<p align="center">
<img src="https://user-images.githubusercontent.com/24568452/204458299-25e305a6-7cbb-4308-86ee-03f146bb938e.png">
</p>

3. 如果更新模型的输入/输出 Tensor 个数、shape 和 Dtype
   发生更新，需要更新文件 `TTSAndroid/app/src/main/java/com/baidu/paddle/lite/demo/tts/Predictor.java`。

### 更新输入

**本 Demo 不包含文本前端模块**，通过下拉框选择预先设置好的文本，在代码中映射成对应的 phone_id，**如需文本前端模块请自行处理**，`phone_id_map.txt`
请参考 [fastspeech2_cnndecoder_csmsc_pdlite_1.3.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_pdlite_1.3.0.zip)。

## 通过 setting 界面更新语音合成的相关参数

### setting 界面参数介绍

可通过 APP 上的 Settings 按钮，实现语音合成 Demo 中参数的更新，目前支持以下参数的更新：
参数的默认值可在 `app/src/main/res/values/strings.xml` 查看

- CPU setting：
    - power_mode 默认是 `LITE_POWER_HIGH`
    - thread_num 默认是 1

### setting 界面参数更新

1. 打开 APP，点击右上角的 `:` 符合，选择 `Settings..` 选项，打开 setting 界面；
2. 再将 setting 界面的 Enable custom settings 选中☑️，然后更新部分参数；
3. 假设更新线程数据，将 CPU Thread Num 设置为 4，更新后，返回原界面，APP 将自动重新加载模型，在下拉框中选择文本会进行合成，合成结束后悔打印 4 线程的耗时和结果

## 性能优化方法

如果你觉得当前性能不符合需求，想进一步提升模型性能，可参考[性能优化文档](https://github.com/PaddlePaddle/Paddle-Lite-Demo#%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96)完成性能优化。

## Release

[2022-11-29-app-release.apk](https://paddlespeech.bj.bcebos.com/demos/TTSAndroid/2022-11-29-app-release.apk)

## More
本 Demo 合并自 [yt605155624/TTSAndroid](https://github.com/yt605155624/TTSAndroid)。
