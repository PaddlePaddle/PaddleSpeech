python analysis.py \
    --filename "recoder_mp_bs16_fp32_ngpu8.txt" \
    --keyword "ips[sent./sec]:" \
    --base_batch_size 16 \
    --model_name "Conformer" \
    --mission_name "eight gpu" \
    --run_mode "mp" \
    --ips_unit "sent./sec" \
    --gpu_num 8 \
    --use_num 480 \
    --separator " " \

