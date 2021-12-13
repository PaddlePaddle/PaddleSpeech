Audio Sample 
==================

The main processes of TTS include:

1. Convert the original text into characters/phonemes, through ``text frontend`` module.

2. Convert characters/phonemes into acoustic features , such as linear spectrogram, mel spectrogram, LPC features, etc. through ``Acoustic models``.

3. Convert acoustic features into waveforms through ``Vocoders``.

When training ``Tacotron2``、``TransformerTTS`` and ``WaveFlow``, we use English single speaker TTS dataset `LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`_  by default. However, when training ``SpeedySpeech``, ``FastSpeech2`` and ``ParallelWaveGAN``, we use Chinese single speaker dataset `CSMSC <https://test.data-baker.com/data/index/source/>`_ by default. 

In the future, ``PaddleSpeech TTS`` will mainly use Chinese TTS datasets for default examples.

Here, we will display three types of audio samples:

1. Analysis/synthesis (ground-truth spectrograms + Vocoder)

2. TTS (Acoustic model + Vocoder)

3. Chinese TTS with/without text frontend (mainly tone sandhi)

Analysis/synthesis
--------------------------

Audio samples generated from ground-truth spectrograms with a vocoder.

.. raw:: html
     
    <b>LJSpeech(English)</b>
    <br>
    </br>
    
    <div class="table">
    <table border="2" cellspacing="1" cellpadding="1"> 
        <tr>
            <th align="center"> Text </th>
            <th align="center"> GT </th>
            <th align="center"> WaveFlow </th>
        </tr>
        <tr>
            <td >Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/ljspeech_gt/LJ001-0001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                
            
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_0.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>in being comparatively modern.</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/ljspeech_gt/LJ001-0002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>

            </td>
            <td>
             <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_1.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
            </audio>
            </td>
        </tr>
        <tr>
            <td>For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/ljspeech_gt/LJ001-0003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_2.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>produced the block books, which were the immediate predecessors of the true printed book</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/ljspeech_gt/LJ001-0004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_3.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing.</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/ljspeech_gt/LJ001-0005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_4.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
    </table>
    
    <br>
    </br>

    <b>CSMSC(Chinese)</b>
    <br>
    </br>

    <table border="2" cellspacing="1" cellpadding="1">
        <tr>
            <th align="center"> Text </th>
            <th align="center"> GT (convert to 24k) </th>
            <th align="center"> ParallelWaveGAN </th>
        </tr>
        <tr>
            <td>昨日，这名“伤者”与医生全部被警方依法刑事拘留</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/baker_gt_24k/009901.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/pwg_baker_ckpt_0.4/009901.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>钱伟长想到上海来办学校是经过深思熟虑的。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/baker_gt_24k/009902.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/pwg_baker_ckpt_0.4/009902.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>她见我一进门就骂，吃饭时也骂，骂得我抬不起头。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/baker_gt_24k/009903.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/pwg_baker_ckpt_0.4/009903.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>李述德在离开之前，只说了一句“柱驼杀父亲了”</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/baker_gt_24k/009904.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/pwg_baker_ckpt_0.4/009904.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>

        </tr>
        <tr>
            <td>这种车票和保险单捆绑出售属于重复性购买。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/baker_gt_24k/009905.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/pwg_baker_ckpt_0.4/009905.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>  
        </tr>    
    </table>
    </div>
    <br>
    <br>

TTS
-------------------

Audio samples generated by a TTS system. Text is first transformed into spectrogram by a text-to-spectrogram model, then the spectrogram is converted into raw audio by a vocoder.

.. raw:: html

    <b>LJSpeech(English)</b>
    <br>
    </br>
    <div class="table">
    <table border="2" cellspacing="1" cellpadding="1"> 
        <tr>
            <th align="center"> Text </th>
            <th align="center"> TransformerTTS + WaveFlow </th>
            <th align="center"> Tacotron2 + WaveFlow </th>
        </tr>
        <tr>
            <td>Life was like a box of chocolates, you never know what you're gonna get.</td>
            <td>
                <audio controls="controls">
                        <source
                            src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/001.wav"
                            type="audio/wav">
                        Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td> 
                <audio controls="controls">
                        <source
                            src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_1.wav"
                            type="audio/wav">
                        Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>With great power there must come great responsibility.</td>
            <td>
                <audio controls="controls">
                        <source
                            src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/002.wav"
                            type="audio/wav">
                        Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td> 
            <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_2.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>To be or not to be, that’s a question.</td>
            <td>
            <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>

            <td> 
            <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_3.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>

        <tr>
            <td>A man can be destroyed but not defeated.</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>

            <td> 
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_4.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>Do not, for one repulse, give up the purpose that you resolved to effort.</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>

            <td> 
            <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_5.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>Death is just a part of life, something we're all destined to do.</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>

            <td> 
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_6.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>I think it's hard winning a war with words. </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>

            <td> 
            <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_7.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>Don’t argue with the people of strong determination, because they may change the fact!</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>

            <td> 
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_8.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>Love you three thousand times.</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>

            <td> 
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_9.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>   
    </table>
    
    <br>
    </br>

    <b>CSMSC(Chinese)</b>
    <br>
    </br>

    <table border="2" cellspacing="1" cellpadding="1"> 
        <tr>
            <th align="center"> Text </th>
            <th align="center"> SpeedySpeech + ParallelWaveGAN </th>
            <th align="center"> FastSpeech2 + ParallelWaveGAN </th>
        </tr>
        <tr>
            <td>凯莫瑞安联合体的经济崩溃，迫在眉睫。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>对于所有想要离开那片废土，去寻找更美好生活的人来说。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>克哈，是你们所有人安全的港湾。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>

        <tr>
            <td>为了保护尤摩扬人民不受异虫的残害，我所做的，比他们自己的领导委员会都多。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>无论他们如何诽谤我，我将继续为所有泰伦人的最大利益，而努力奋斗。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>身为你们的元首，我带领泰伦人实现了人类统治领地和经济的扩张。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>我们将继续成长，用行动回击那些只会说风凉话，不愿意和我们相向而行的害群之马。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>帝国武装力量，无数的优秀儿女，正时刻守卫着我们的家园大门，但是他们孤木难支。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>凡是今天应征入伍者，所获的所有刑罚罪责，减半。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>   
    </table>

    <br>
    </br>

    <table border="2" cellspacing="1" cellpadding="1"> 
        <tr>
            <th align="center"> FastSpeech2-Conformer + ParallelWaveGAN </th>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_conformer_baker_ckpt_0.5_pwg_baker_ckpt_0.4/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_conformer_baker_ckpt_0.5_pwg_baker_ckpt_0.4/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_conformer_baker_ckpt_0.5_pwg_baker_ckpt_0.4/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>

        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_conformer_baker_ckpt_0.5_pwg_baker_ckpt_0.4/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_conformer_baker_ckpt_0.5_pwg_baker_ckpt_0.4/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_conformer_baker_ckpt_0.5_pwg_baker_ckpt_0.4/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_conformer_baker_ckpt_0.5_pwg_baker_ckpt_0.4/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_conformer_baker_ckpt_0.5_pwg_baker_ckpt_0.4/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_conformer_baker_ckpt_0.5_pwg_baker_ckpt_0.4/009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>   
    </table>
    </div>
    <br>
    <br>


Multi-Speaker TTS
-------------------

PaddleSpeech also support Multi-Speaker TTS, we provide the audio demos generated by FastSpeech2 + ParallelWaveGAN, we use AISHELL-3 Multi-Speaker TTS dataset. Each line is a different person.


.. raw:: html

    <div class="table">
    <table border="2" cellspacing="1" cellpadding="1">
        <tr>
            <th align="center"> Target Timbre </th>
            <th align="center"> Generated </th>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/0.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/0_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/1.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/1_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/2.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/2_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/3.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/3_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/4.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/4_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/5.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/5_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/6.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/6_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/7.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/7_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/8.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/8_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/9.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/9_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/10.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/10_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/11.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/11_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/12.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/12_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/13.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/13_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/14.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/14_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/15.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/15_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/16.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/16_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/17.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/17_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/18.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/18_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/target/19.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fs2_aishell3_demos/generated/19_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>

    <table>
    <div>
    <br>
    <br>
        

Style control in FastSpeech2
--------------------------------------
In our FastSpeech2, we can control ``duration``, ``pitch`` and ``energy``.

We provide the audio demos of duration control here. ``duration`` means the duration of phonemes, when we reduce duration, the speed of audios will increase, and when we incerase ``duration``, the speed of audios will reduce.

The ``duration`` of different phonemes in a sentence can have different scale ratios (when you want to slow down one word and keep the other words' speed in a sentence). Here we use a fixed scale ratio for different phonemes to control the ``speed`` of audios.

The duration control in FastSpeech2 can control the speed of audios will keep the pitch. (in some speech tool, increase the speed will increase the pitch, and vice versa.)

.. raw:: html

    <div class="table">
    <table border="2" cellspacing="1" cellpadding="1">
        <tr>
            <th align="center"> Speed(0.8x) </th>
            <th align="center"> Speed(1x) </th>
            <th align="center"> Speed(1.2x) </th>
        </tr>
        <tr>
             <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x0.8_001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1_001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1.2_001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
             <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x0.8_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1.2_002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
             <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x0.8_003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1_003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1.2_003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
             <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x0.8_004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1_004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1.2_004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
             <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x0.8_005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1_005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1.2_005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
             <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x0.8_007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1_007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1.2_007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
             <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x0.8_008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1_008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1.2_008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
             <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x0.8_009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1_009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speed/x1.2_009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>

    <table>
    <div>
    <br>
    <br>

We provide the audio demos of pitch control here. 

When we set pitch of one sentence to a mean value and set ``tones`` of phones to ``1``, we will get a ``robot-style`` timbre.

When we raise the pitch of an adult female (with a fixed scale ratio), we will get a ``child-style`` timbre.

The ``pitch`` of different phonemes in a sentence can also have different scale ratios.

The nomal audios are in the second column of the previous table.

.. raw:: html

    <div class="table">
    <table border="2" cellspacing="1" cellpadding="1">
        <tr>
            <th align="center"> Robot </th>
            <th align="center"> Child </th>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/robot/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/child_voice/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/robot/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/child_voice/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/robot/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/child_voice/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/robot/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/child_voice//004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/robot/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/child_voice//005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/robot/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/child_voice//007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/robot/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/child_voice//008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/robot/009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/child_voice//009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>

    <table>
    <div>
    <br>
    <br>


Chinese TTS with/without text frontend
--------------------------------------

We provide a complete Chinese text frontend module in ``PaddleSpeech TTS``. ``Text Normalization`` and ``G2P`` are the most important modules in text frontend, We assume that the texts are normalized already, and mainly compare ``G2P`` module here.

We use ``FastSpeech2`` + ``ParallelWaveGAN`` here.

.. raw:: html

    <div class="table">
    <table border="2" cellspacing="1" cellpadding="1">
        <tr>
            <th align="center"> Text</th>
            <th align="center"> With Text Frontend </th>
            <th align="center"> Without Text Frontend </th>
        </tr>
        <tr>
            <td>他只是一个纸老虎。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>手表厂有五种好产品。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>老板的轿车需要保养。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>我们所有人都好喜欢你呀。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>岂有此理。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>虎骨酒多少钱一瓶。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>这件事情需要冷处理。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>这个老奶奶是个大喇叭。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>我喜欢说相声。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
        <tr>
            <td>有一天，我路过了一栋楼。</td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/010.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/010.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>

    <table>
    </div>
    <br>
    <br> 

   