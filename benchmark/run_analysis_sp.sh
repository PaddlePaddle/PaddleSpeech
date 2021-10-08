python analysis.py \
    --filename "recoder_sp_bs16_fp32_ngpu1.txt" \
    --keyword "ips[sent./sec]:" \
    --base_batch_size 16 \
    --model_name "Conformer" \
    --mission_name "one gpu" \
    --run_mode "sp" \
    --ips_unit "sent./sec" \
    --gpu_num 1 \
    --use_num 60 \
    --separator " " \

