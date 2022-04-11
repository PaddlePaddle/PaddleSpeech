model_path=/home/users/liangyunming/.paddlespeech/models/
#am_model_dir=$model_path/fastspeech2_csmsc-zh/fastspeech2_nosil_baker_ckpt_0.4/    ## fastspeech2
am_model_dir=$model_path/fastspeech2_csmsc-zh/fastspeech2_cnndecoder_csmsc_ckpt_1.0.0/    ## fastspeech2_cnn
voc_model_dir=$model_path/hifigan_csmsc-zh/hifigan_csmsc_ckpt_0.1.1/    ## hifigan
#voc_model_dir=$model_path/mb_melgan_csmsc-zh/mb_melgan_csmsc_ckpt_0.1.1/    ## mb_melgan

if [[ $am_model_dir == *"fastspeech2_cnndecoder"* ]]; then
    am_support_stream=True
else
    am_support_stream=False
fi

# get am file
for file in $(ls $am_model_dir)
do
    if [[ $file == *"yaml"* ]]; then
        am_config_file=$file
    elif [[ $file == *"pdz"* ]]; then
        am_ckpt_file=$file
    elif [[ $file == *"stat"* ]]; then
        am_stat_file=$file
    elif [[ $file == *"phone"* ]]; then
        phones_dict_file=$file
    fi
    
done

# get voc file
for file in $(ls $voc_model_dir)
do
    if [[ $file == *"yaml"* ]]; then
        voc_config_file=$file
    elif [[ $file == *"pdz"* ]]; then
        voc_ckpt_file=$file
    elif [[ $file == *"stat"* ]]; then
        voc_stat_file=$file
    fi
    
done


#run
python test_online_tts.py --am fastspeech2_csmsc \
                          --am_support_stream $am_support_stream \
                          --am_config $am_model_dir/$am_config_file \
                          --am_ckpt $am_model_dir/$am_ckpt_file \
                          --am_stat $am_model_dir/$am_stat_file \
                          --phones_dict $am_model_dir/$phones_dict_file \
                          --voc hifigan_csmsc \
                          --voc_config $voc_model_dir/$voc_config_file \
                          --voc_ckpt $voc_model_dir/$voc_ckpt_file \
                          --voc_stat $voc_model_dir/$voc_stat_file  \
                          --lang zh \
                          --device cpu \
                          --text ./csmsc_test.txt \
                          --output_dir ./output \
                          --log_file ./result.log \
                          --am_streaming False \
                          --am_pad 12 \
                          --am_block 42 \
                          --voc_streaming True \
                          --voc_pad 14 \
                          --voc_block 14 \

