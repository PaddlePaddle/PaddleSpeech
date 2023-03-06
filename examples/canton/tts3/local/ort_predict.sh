train_output_path=$1

stage=0
stop_stage=0

# e2e, synthesize from text
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../ort_predict_e2e.py \
        --inference_dir=${train_output_path}/inference_onnx \
        --am=fastspeech2_canton \
        --voc=pwgan_aishell3 \
        --spk_id=10 \
        --output_dir=${train_output_path}/onnx_infer_out_e2e \
        --text=${BIN_DIR}/../sentences_canton.txt \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --lang=canton \
        --device=cpu \
        --cpu_threads=2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/../ort_predict_e2e.py \
        --inference_dir=${train_output_path}/inference_onnx \
        --am=fastspeech2_canton \
        --voc=mb_melgan_csmsc \
        --spk_id=10 \
        --output_dir=${train_output_path}/onnx_infer_out_e2e \
        --text=${BIN_DIR}/../sentences_canton.txt \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --lang=canton \
        --device=cpu \
        --cpu_threads=2
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python3 ${BIN_DIR}/../ort_predict_e2e.py \
        --inference_dir=${train_output_path}/inference_onnx \
        --am=fastspeech2_canton \
        --voc=hifigan_csmsc \
        --output_dir=${train_output_path}/onnx_infer_out_e2e \
        --text=${BIN_DIR}/../sentences_canton.txt \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --lang=canton \
        --device=cpu \
        --cpu_threads=2
fi

# synthesize from metadata, take hifigan as an example
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python3 ${BIN_DIR}/../ort_predict.py \
        --inference_dir=${train_output_path}/inference_onnx \
        --am=fastspeech2_canton \
        --voc=hifigan_csmsc \
        --test_metadata=dump/test/norm/metadata.jsonl \
        --output_dir=${train_output_path}/onnx_infer_out \
        --device=cpu \
        --cpu_threads=2
fi