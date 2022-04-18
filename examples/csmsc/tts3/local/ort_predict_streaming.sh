train_output_path=$1

stage=0
stop_stage=0

# e2e, synthesize from text
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../ort_predict_streaming.py \
        --inference_dir=${train_output_path}/inference_onnx_streaming \
        --am=fastspeech2_csmsc \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_csmsc \
        --output_dir=${train_output_path}/onnx_infer_out_streaming \
        --text=${BIN_DIR}/../csmsc_test.txt \
        --phones_dict=dump/phone_id_map.txt \
        --device=cpu \
        --cpu_threads=2 \
        --am_streaming=True
fi
