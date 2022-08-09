train_output_path=$1

stage=0
stop_stage=0

# e2e, synthesize from text
# voc: pwgan_aishell3
# the spk_id=174 means baker speaker, default
# the spk_id=175 means ljspeech speaker
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../ort_predict_e2e.py \
        --inference_dir=${train_output_path}/inference_onnx \
        --am=fastspeech2_mix \
        --voc=pwgan_aishell3 \
        --output_dir=${train_output_path}/onnx_infer_out_e2e \
        --text=${BIN_DIR}/../sentences_mix.txt \
        --phones_dict=dump/phone_id_map.txt \
        --device=cpu \
        --cpu_threads=4 \
        --lang=mix \
        --spk_id=174 
        

fi


# voc: hifigan_aishell3
# the spk_id=174 means baker speaker, default
# the spk_id=175 means ljspeech speaker
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/../ort_predict_e2e.py \
        --inference_dir=${train_output_path}/inference_onnx \
        --am=fastspeech2_mix \
        --voc=hifigan_aishell3 \
        --output_dir=${train_output_path}/onnx_infer_out_e2e \
        --text=${BIN_DIR}/../sentences_mix.txt \
        --phones_dict=dump/phone_id_map.txt \
        --device=cpu \
        --cpu_threads=4 \
        --lang=mix \
        --spk_id=174 
        
fi
