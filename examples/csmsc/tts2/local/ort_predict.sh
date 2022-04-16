train_output_path=$1

stage=0
stop_stage=0

# only support default_fastspeech2/speedyspeech + hifigan/mb_melgan now!

# synthesize from metadata
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../ort_predict.py \
        --inference_dir=${train_output_path}/inference_onnx \
        --am=speedyspeech_csmsc \
        --voc=hifigan_csmsc \
        --test_metadata=dump/test/norm/metadata.jsonl \
        --output_dir=${train_output_path}/onnx_infer_out \
        --device=cpu \
        --cpu_threads=2
fi

# e2e, synthesize from text
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/../ort_predict_e2e.py \
        --inference_dir=${train_output_path}/inference_onnx \
        --am=speedyspeech_csmsc \
        --voc=hifigan_csmsc \
        --output_dir=${train_output_path}/onnx_infer_out_e2e \
        --text=${BIN_DIR}/../csmsc_test.txt \
        --phones_dict=dump/phone_id_map.txt \
        --tones_dict=dump/tone_id_map.txt \
        --device=cpu \
        --cpu_threads=2
fi
