# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import wave

import numpy as np


def wav2pcm(wavfile, pcmfile, data_type=np.int16):
    with open(wavfile, "rb") as f:
        f.seek(0)
        f.read(44)
        data = np.fromfile(f, dtype=data_type)
        data.tofile(pcmfile)


def pcm2wav(pcm_file, wav_file, channels=1, bits=16, sample_rate=16000):
    pcmf = open(pcm_file, 'rb')
    pcmdata = pcmf.read()
    pcmf.close()

    if bits % 8 != 0:
        raise ValueError("bits % 8 must == 0. now bits:" + str(bits))

    wavfile = wave.open(wav_file, 'wb')
    wavfile.setnchannels(channels)
    wavfile.setsampwidth(bits // 8)
    wavfile.setframerate(sample_rate)
    wavfile.writeframes(pcmdata)
    wavfile.close()


def change_speed(sample_raw, speed_rate, sample_rate):
    """Change the audio speed by linear interpolation.
    Note that this is an in-place transformation.
    :param speed_rate: Rate of speed change:
                       speed_rate > 1.0, speed up the audio;
                       speed_rate = 1.0, unchanged;
                       speed_rate < 1.0, slow down the audio;
                       speed_rate <= 0.0, not allowed, raise ValueError.
    :type speed_rate: float
    :raises ValueError: If speed_rate <= 0.0.
    """
    if speed_rate == 1.0:
        return sample_raw
    if speed_rate <= 0:
        raise ValueError("speed_rate should be greater than zero.")

    # numpy
    # old_length = self._samples.shape[0]
    # new_length = int(old_length / speed_rate)
    # old_indices = np.arange(old_length)
    # new_indices = np.linspace(start=0, stop=old_length, num=new_length)
    # self._samples = np.interp(new_indices, old_indices, self._samples)

    # sox, slow
    try:
        import soxbindings as sox
    except ImportError:
        try:
            from paddlespeech.s2t.utils import dynamic_pip_install
            package = "sox"
            dynamic_pip_install.install(package)
            package = "soxbindings"
            dynamic_pip_install.install(package)
            import soxbindings as sox
        except Exception:
            raise RuntimeError("Can not install soxbindings on your system.")

    tfm = sox.Transformer()
    tfm.set_globals(multithread=False)
    tfm.tempo(speed_rate)
    sample_speed = tfm.build_array(
        input_array=sample_raw,
        sample_rate_in=sample_rate).squeeze(-1).astype(np.float32).copy()

    return sample_speed
