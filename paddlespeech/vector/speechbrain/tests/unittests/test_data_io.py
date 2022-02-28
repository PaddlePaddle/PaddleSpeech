import paddle
import os
import numpy as np

def test_read_audio(tmpdir, device):
    from speechbrain.dataio.dataio import read_audio, write_audio
    from paddleaudio.backends.audio import depth_convert
    paddle.device.set_device("cpu")
    test_waveform = paddle.clip(paddle.rand([16000]), min=-1, max=1)
    wavfile = os.path.join(tmpdir, "wave.wav")
    write_audio(wavfile, test_waveform.cpu(), 16000)
    # dummy annotation
    for i in range(3):
        start = paddle.randint(0, 8000, (1,)).item()
        stop = start + paddle.randint(500, 1000, (1,)).item()
        wav_obj = {"wav": {"file": wavfile, "start": start, "stop": stop}}
        loaded = paddle.to_tensor(read_audio(wav_obj["wav"]))
        assert loaded.allclose(test_waveform[start:stop], atol=1e-4)
        # set to equal when switching to the sox_io backend

        loaded = paddle.to_tensor(read_audio(wavfile))
        assert loaded.allclose(test_waveform, atol=1e-4)

def test_read_audio_multichannel(tmpdir, device):
    from speechbrain.dataio.dataio import read_audio_multichannel, write_audio
    paddle.device.set_device("cpu")
    test_waveform = paddle.clip(paddle.rand([16000, 2]), min=-1, max=1)
    wavfile = os.path.join(tmpdir, "wave.wav")
    # sf.write(wavfile, test_waveform, 16000, subtype="float")
    write_audio(wavfile, test_waveform.cpu(), 16000)

    # dummy annotation we save and load one multichannel file
    for i in range(2):
        start = paddle.randint(0, 8000, (1,)).item()
        stop = start + paddle.randint(500, 1000, (1,)).item()

        wav_obj = {"wav": {"files": [wavfile], "start": start, "stop": stop}}

        loaded = paddle.to_tensor(read_audio_multichannel(wav_obj["wav"]))
        assert loaded.allclose(test_waveform[start:stop, :], atol=1e-4)
        # set to equal when switching to the sox_io backend

        loaded = paddle.to_tensor(read_audio_multichannel(wavfile))
        assert loaded.allclose(test_waveform, atol=1e-4)
    # we test now multiple files loading
    test_waveform_2 = paddle.clip(paddle.rand([16000, 2]), min=-1, max=1)
    wavfile_2 = os.path.join(tmpdir, "wave_2.wav")
    write_audio(wavfile_2, test_waveform_2.cpu(), 16000)
    # sf.write(wavfile_2, test_waveform_2, 16000, subtype="float")
    for i in range(2):
        start = paddle.randint(0, 8000, (1,)).item()
        stop = start + paddle.randint(500, 1000, (1,)).item()
        wav_obj = {
            "wav": {"files": [wavfile, wavfile_2], "start": start, "stop": stop}
        }
        loaded = paddle.to_tensor(read_audio_multichannel(wav_obj["wav"]))
        test_waveform3 = paddle.concat(
            (test_waveform[start:stop, :], test_waveform_2[start:stop, :]), axis=1
        )
        assert loaded[1].allclose(test_waveform3[1], atol=1e-4)

