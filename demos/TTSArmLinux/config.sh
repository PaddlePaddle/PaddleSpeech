# configuration

ARM_ABI=armv8
#ARM_ABI=armv7hf

MODELS_DIR="${PWD}/models"
LIBS_DIR="${PWD}/libs"

PADDLE_LITE_DOWNLOAD_URL="https://github.com/SwimmingTiger/Paddle-Lite/releases/download/68b66fd35/inference_lite_lib.armlinux.${ARM_ABI}.gcc.with_extra.with_cv.tar.gz"
PADDLE_LITE_DIR="${LIBS_DIR}/inference_lite_lib.armlinux.${ARM_ABI}.gcc.with_extra.with_cv/cxx"

MODEL_DOWNLOAD_URL="https://paddlespeech.bj.bcebos.com/demos/TTSAndroid/fs2cnn_mbmelgan_cpu_v1.3.0.tar.gz"
AM_MODEL_PATH="${MODELS_DIR}/cpu/fastspeech2_csmsc_arm.nb"
VOC_MODEL_PATH="${MODELS_DIR}/cpu/mb_melgan_csmsc_arm.nb"
