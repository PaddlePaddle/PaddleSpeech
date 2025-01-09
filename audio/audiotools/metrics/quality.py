# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/metrics/quality.py)
import os

import numpy as np
import paddle

from ..core import AudioSignal


def visqol(
        estimates: AudioSignal,
        references: AudioSignal,
        mode: str="audio", ):
    """ViSQOL score.

    Parameters
    ----------
    estimates : AudioSignal
        Degraded AudioSignal
    references : AudioSignal
        Reference AudioSignal
    mode : str, optional
        'audio' or 'speech', by default 'audio'

    Returns
    -------
    Tensor[float]
        ViSQOL score (MOS-LQO)
    """
    try:
        from pyvisqol import visqol_lib_py
        from pyvisqol.pb2 import visqol_config_pb2
        from pyvisqol.pb2 import similarity_result_pb2
    except ImportError:
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2
        from visqol.pb2 import similarity_result_pb2

    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        target_sr = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        target_sr = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.audio.sample_rate = target_sr
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    estimates = estimates.clone().to_mono().resample(target_sr)
    references = references.clone().to_mono().resample(target_sr)

    visqols = []
    for i in range(estimates.batch_size):
        _visqol = api.Measure(
            references.audio_data[i, 0].detach().cpu().numpy().astype(float),
            estimates.audio_data[i, 0].detach().cpu().numpy().astype(float), )
        visqols.append(_visqol.moslqo)
    return paddle.to_tensor(np.array(visqols))


if __name__ == "__main__":
    signal = AudioSignal(paddle.randn([44100]), 44100)
    print(visqol(signal, signal))
