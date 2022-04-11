model_path=~/.paddlespeech/models/
am_model_dir=$model_path/fastspeech2_csmsc-zh/fastspeech2_cnndecoder_csmsc_ckpt_1.0.0/    ## fastspeech2_c
voc_model_dir=$model_path/mb_melgan_csmsc-zh/mb_melgan_csmsc_ckpt_0.1.1/    ## mb_melgan
testdata=../../../../t2s/exps/csmsc_test.txt

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


# run test
# am can choose fastspeech2_csmsc or fastspeech2-C_csmsc, where fastspeech2-C_csmsc supports streaming inference.
# voc can choose hifigan_csmsc and mb_melgan_csmsc, They can both support streaming inference.
python test_online_tts.py --am fastspeech2-C_csmsc \
                          --am_config $am_model_dir/$am_config_file \
                          --am_ckpt $am_model_dir/$am_ckpt_file \
                          --am_stat $am_model_dir/$am_stat_file \
                          --phones_dict $am_model_dir/$phones_dict_file \
                          --voc mb_melgan_csmsc \
                          --voc_config $voc_model_dir/$voc_config_file \
                          --voc_ckpt $voc_model_dir/$voc_ckpt_file \
                          --voc_stat $voc_model_dir/$voc_stat_file  \
                          --lang zh \
                          --device cpu \
                          --text $testdata \
                          --output_dir ./output \
                          --log_file ./result.log \
                          --am_streaming True \
                          --am_pad 12 \
                          --am_block 42 \
                          --voc_streaming True \
                          --voc_pad 14 \
                          --voc_block 14 \

