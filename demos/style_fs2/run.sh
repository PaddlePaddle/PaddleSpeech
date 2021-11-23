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
    # download pretrained tts models and unzip
    wget -P download https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip
    unzip -d download download/pwg_baker_ckpt_0.4.zip
    wget -P download https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip
    unzip -d download download/fastspeech2_nosil_baker_ckpt_0.4.zip
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # run tts
    CUDA_VISIBLE_DEVICES=${gpus} \
    python3 style_syn.py \
        --fastspeech2-config=download/fastspeech2_nosil_baker_ckpt_0.4/default.yaml \
        --fastspeech2-checkpoint=download/fastspeech2_nosil_baker_ckpt_0.4/snapshot_iter_76000.pdz \
        --fastspeech2-stat=download/fastspeech2_nosil_baker_ckpt_0.4/speech_stats.npy \
        --fastspeech2-pitch-stat=download/fastspeech2_nosil_baker_ckpt_0.4/pitch_stats.npy \
        --fastspeech2-energy-stat=download/fastspeech2_nosil_baker_ckpt_0.4/energy_stats.npy \
        --pwg-config=download/pwg_baker_ckpt_0.4/pwg_default.yaml \
        --pwg-checkpoint=download/pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
        --pwg-stat=download/pwg_baker_ckpt_0.4/pwg_stats.npy \
        --text=${BIN_DIR}/../sentences.txt \
        --output-dir=output \
        --phones-dict=download/fastspeech2_nosil_baker_ckpt_0.4/phone_id_map.txt
fi
