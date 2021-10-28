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
    <table>
        <tr>
            <th  align="left"> GT </th>
            <th  align="left"> WaveFlow </th>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/ljspeech_gt/LJ001-0001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/ljspeech_gt/LJ001-0002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/ljspeech_gt/LJ001-0003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/ljspeech_gt/LJ001-0004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
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
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_0.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_1.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_2.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_3.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_4.wav"
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

    <table>
        <tr>
            <th  align="left"> GT (convert to 24k) </th>
            <th  align="left"> ParallelWaveGAN </th>
        </tr>
        <tr>
           <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/baker_gt_24k/009901.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/baker_gt_24k/009902.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/baker_gt_24k/009903.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/baker_gt_24k/009904.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
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
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/pwg_baker_ckpt_0.4/009901.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/pwg_baker_ckpt_0.4/009902.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/pwg_baker_ckpt_0.4/009903.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/pwg_baker_ckpt_0.4/009904.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/pwg_baker_ckpt_0.4/009905.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
    
    </table>


TTS
-------------------

Audio samples generated by a TTS system. Text is first transformed into spectrogram by a text-to-spectrogram model, then the spectrogram is converted into raw audio by a vocoder.

.. raw:: html

    <table>
        <tr>
            <th  align="left"> TransformerTTS + WaveFlow </th>
            <th  align="left"> Tacotron2 + WaveFlow </th>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/transformer_tts_ljspeech_ckpt_0.4_waveflow_ljspeech_ckpt_0.3/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
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
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_waveflow_samples_0.2/sentence_1.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_waveflow_samples_0.2/sentence_2.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_waveflow_samples_0.2/sentence_3.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_waveflow_samples_0.2/sentence_4.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_waveflow_samples_0.2/sentence_5.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_waveflow_samples_0.2/sentence_6.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_waveflow_samples_0.2/sentence_7.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_waveflow_samples_0.2/sentence_8.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_waveflow_samples_0.2/sentence_9.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
    </table>

    <table>
        <tr>
            <th  align="left"> SpeedySpeech + ParallelWaveGAN </th>
            <th  align="left"> FastSpeech2 + ParallelWaveGAN </th>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/speedyspeech_baker_ckpt_0.4_pwg_baker_ckpt_0.4/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
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
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/fastspeech2_nosil_baker_ckpt_0.4_parallel_wavegan_baker_ckpt_0.4/009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>
    </table>



Chinese TTS with/without text frontend
--------------------------------------

We provide a complete Chinese text frontend module in ``PaddleSpeech TTS``. ``Text Normalization`` and ``G2P`` are the most important modules in text frontend, We assume that the texts are normalized already, and mainly compare ``G2P`` module here.

We use ``FastSpeech2`` + ``ParallelWaveGAN`` here.

.. raw:: html

    <table>
        <tr>
            <th  align="left"> With Text Frontend </th>
            <th  align="left"> Without Text Frontend </th>
        </tr>
        <tr>
            <td>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/with_frontend/009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
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
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/001.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/002.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/003.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/004.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/005.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/006.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/007.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/008.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/009.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
                <audio controls="controls">
                    <source
                        src="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/without_frontend/010.wav"
                        type="audio/wav">
                    Your browser does not support the <code>audio</code> element.
                </audio>
            </td>
        </tr>


    <table>