#!/bin/bash
set -e
echo -e "\e[1;31monly if you see 'Test success !!!', the cli testing is successful\e[0m"

# Audio classification
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav
paddlespeech cls --input ./cat.wav --topk 10

# Punctuation_restoration
paddlespeech text --input 今天的天气真不错啊你下午有空吗我想约你一起去吃饭 --model ernie_linear_p3_wudao_fast

# Speech SSL
paddlespeech ssl --task asr --lang en --input ./en.wav
paddlespeech ssl --task vector --lang en --input ./en.wav

# Speech_recognition
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/ch_zh_mix.wav
paddlespeech asr --input ./zh.wav
paddlespeech asr --model conformer_aishell --input ./zh.wav
paddlespeech asr --model conformer_online_aishell --input ./zh.wav
paddlespeech asr --model conformer_online_wenetspeech --input ./zh.wav
paddlespeech asr --model conformer_online_multicn --input ./zh.wav
paddlespeech asr --model conformer_u2pp_online_wenetspeech --lang zh --input zh.wav
paddlespeech asr --model transformer_librispeech --lang en --input ./en.wav
paddlespeech asr --model deepspeech2offline_aishell --input ./zh.wav
paddlespeech asr --model deepspeech2online_wenetspeech --input ./zh.wav
paddlespeech asr --model deepspeech2online_aishell --input ./zh.wav
paddlespeech asr --model deepspeech2offline_librispeech --lang en --input ./en.wav
paddlespeech asr --model conformer_talcs --lang zh_en --codeswitch True --input ./ch_zh_mix.wav

# Support editing num_decoding_left_chunks
paddlespeech asr --model conformer_online_wenetspeech --num_decoding_left_chunks 3 --input ./zh.wav

# long audio restriction
{
wget -c https://paddlespeech.bj.bcebos.com/datasets/single_wav/zh/test_long_audio_01.wav
paddlespeech asr --model deepspeech2online_wenetspeech --input test_long_audio_01.wav -y
if [ $? -ne 255 ]; then
   echo -e "\e[1;31mTime restriction not passed\e[0m"
   exit 1
fi
} &&
{
 echo -e "\033[32mTime restriction passed\033[0m"
}

# Text To Speech
paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --am speedyspeech_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --voc mb_melgan_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --voc style_melgan_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --voc pwgan_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --am fastspeech2_aishell3 --voc pwgan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0
paddlespeech tts --am fastspeech2_aishell3 --voc hifigan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0
paddlespeech tts --am fastspeech2_ljspeech --voc pwgan_ljspeech --lang en --input "Life was like a box of chocolates, you never know what you're gonna get."
paddlespeech tts --am fastspeech2_ljspeech --voc hifigan_ljspeech --lang en --input "Life was like a box of chocolates, you never know what you're gonna get."
paddlespeech tts --am fastspeech2_vctk --voc pwgan_vctk --input "Life was like a box of chocolates, you never know what you're gonna get." --lang en --spk_id 0
paddlespeech tts --am fastspeech2_vctk --voc hifigan_vctk --input "Life was like a box of chocolates, you never know what you're gonna get." --lang en --spk_id 0
paddlespeech tts --am tacotron2_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --am tacotron2_csmsc --voc wavernn_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --am tacotron2_ljspeech --voc pwgan_ljspeech --lang en --input "Life was like a box of chocolates, you never know what you're gonna get."
paddlespeech tts --am fastspeech2_male --voc pwgan_male --lang zh --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --am fastspeech2_male --voc pwgan_male --lang en --input "Life was like a box of chocolates, you never know what you're gonna get."

# mix tts
# The `am` must be `fastspeech2_mix`!
# The `lang` must be `mix`!
# The voc must be chinese datasets' voc now!
# spk 174 is csmcc, spk 175 is ljspeech
paddlespeech tts --am fastspeech2_mix --voc hifigan_csmsc --lang mix --input "热烈欢迎您在 Discussions 中提交问题，并在 Issues 中指出发现的 bug。此外，我们非常希望您参与到 Paddle Speech 的开发中！" --spk_id 174 --output mix_spk174.wav
paddlespeech tts --am fastspeech2_mix --voc hifigan_aishell3 --lang mix --input "热烈欢迎您在 Discussions 中提交问题，并在 Issues 中指出发现的 bug。此外，我们非常希望您参与到 Paddle Speech 的开发中！" --spk_id 174 --output mix_spk174_aishell3.wav
paddlespeech tts --am fastspeech2_mix --voc pwgan_csmsc --lang mix --input "我们的声学模型使用了 Fast Speech Two, 声码器使用了 Parallel Wave GAN and Hifi GAN." --spk_id 175 --output mix_spk175_pwgan.wav
paddlespeech tts --am fastspeech2_mix --voc hifigan_csmsc --lang mix --input "我们的声学模型使用了 Fast Speech Two, 声码器使用了 Parallel Wave GAN and Hifi GAN." --spk_id 175 --output mix_spk175.wav

# male mix tts
paddlespeech tts --am fastspeech2_male --voc pwgan_male --lang mix --input "我们的声学模型使用了 Fast Speech Two, 声码器使用了 Parallel Wave GAN and Hifi GAN." --output male_mix_fs2_pwgan.wav

# Speech Translation (only support linux)
paddlespeech st --input ./en.wav

# Speaker Verification
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav
paddlespeech vector --task spk --input 85236145389.wav

# batch process
echo -e "1 欢迎光临。\n2 谢谢惠顾。" | paddlespeech tts

echo -e "demo1 85236145389.wav \n demo2 85236145389.wav" > vec.job
paddlespeech vector --task spk --input vec.job

echo -e "demo3 85236145389.wav \n demo4 85236145389.wav" | paddlespeech vector --task spk
rm 85236145389.wav
rm vec.job

# shell pipeline
paddlespeech asr --input ./zh.wav | paddlespeech text --task punc

# stats
paddlespeech stats --task asr
paddlespeech stats --task tts
paddlespeech stats --task cls
paddlespeech stats --task text
paddlespeech stats --task vector
paddlespeech stats --task st

# whisper text recognize
paddlespeech whisper --task transcribe --input ./zh.wav

# whisper recognize text and translate to English
paddlespeech whisper --task translate --input ./zh.wav


echo -e "\033[32mTest success !!!\033[0m"
