#!/bin/bash
source path.sh

gpus=0
stage=0
stop_stage=100

# with the following command, you can choice the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

mkdir -p download

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # install PaddleOCR
    pip install "paddleocr>=2.0.1"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # download pretrained tts models and unzip
    wget -P download https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip
    unzip -d download download/pwg_baker_ckpt_0.4.zip
    wget -P download https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip
    unzip -d download download/fastspeech2_nosil_baker_ckpt_0.4.zip
    # download sources
    wget -P download https://paddlespeech.bj.bcebos.com/demos/story_talker/simfang.ttf
    wget -P download/imgs https://paddlespeech.bj.bcebos.com/demos/story_talker/000.jpg
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # run ocr
    CUDA_VISIBLE_DEVICES=${gpus} \
    python3 ocr.py --img-dir=download/imgs --output-dir=output --font-path=download/simfang.ttf
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # run tts
    CUDA_VISIBLE_DEVICES=${gpus} \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=fastspeech2_csmsc \
        --am_config=download/fastspeech2_nosil_baker_ckpt_0.4/default.yaml \
        --am_ckpt=download/fastspeech2_nosil_baker_ckpt_0.4/snapshot_iter_76000.pdz \
        --am_stat=download/fastspeech2_nosil_baker_ckpt_0.4/speech_stats.npy  \
        --voc=pwgan_csmsc \
        --voc_config=download/pwg_baker_ckpt_0.4/pwg_default.yaml \
        --voc_ckpt=download/pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
        --voc_stat=download/pwg_baker_ckpt_0.4/pwg_stats.npy \
        --lang=zh \
        --text=output/sentences.txt \
        --output_dir=output/wavs \
        --inference_dir=output/inference \
        --phones_dict=download/fastspeech2_nosil_baker_ckpt_0.4/phone_id_map.txt
    # output/inference is not needed here, which save the static models
    rm -rf output/inference
fi
