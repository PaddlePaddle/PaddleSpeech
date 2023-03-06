# configuration

ARM_ABI=armv8
#ARM_ABI=armv7hf

MODELS_DIR="${PWD}/models"
LIBS_DIR="${PWD}/libs"
OUTPUT_DIR="${PWD}/output"

PADDLE_LITE_DIR="${LIBS_DIR}/inference_lite_lib.armlinux.${ARM_ABI}.gcc.with_extra.with_cv/cxx"

AM_MODEL_PATH="${MODELS_DIR}/cpu/fastspeech2_csmsc_arm.nb"
VOC_MODEL_PATH="${MODELS_DIR}/cpu/mb_melgan_csmsc_arm.nb"
