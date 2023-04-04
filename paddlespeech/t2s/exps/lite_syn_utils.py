import os
from pathlib import Path
from typing import Optional

import numpy as np
from paddlelite.lite import create_paddle_predictor
from paddlelite.lite import MobileConfig

from .syn_utils import run_frontend


# Paddle-Lite
def get_lite_predictor(model_dir: Optional[os.PathLike] = None,
                       model_file: Optional[os.PathLike] = None,
                       cpu_threads: int = 1):
    config = MobileConfig()
    config.set_model_from_file(str(Path(model_dir) / model_file))
    predictor = create_paddle_predictor(config)
    return predictor


def get_lite_am_output(input: str,
                       am_predictor,
                       am: str,
                       frontend: object,
                       lang: str = 'zh',
                       merge_sentences: bool = True,
                       speaker_dict: Optional[os.PathLike] = None,
                       spk_id: int = 0,
                       add_blank: bool = False):
    am_name = am[:am.rindex('_')]
    am_dataset = am[am.rindex('_') + 1:]
    get_spk_id = False
    get_tone_ids = False
    if am_name == 'speedyspeech':
        get_tone_ids = True
    if am_dataset in {"aishell3", "vctk", "mix"} and speaker_dict:
        get_spk_id = True
        spk_id = np.array([spk_id])

    frontend_dict = run_frontend(
        frontend=frontend,
        text=input,
        merge_sentences=merge_sentences,
        get_tone_ids=get_tone_ids,
        lang=lang,
        add_blank=add_blank,
    )

    if get_tone_ids:
        tone_ids = frontend_dict['tone_ids']
        tones = tone_ids[0].numpy()
        tones_handle = am_predictor.get_input(1)
        tones_handle.from_numpy(tones)

    if get_spk_id:
        spk_id_handle = am_predictor.get_input(1)
        spk_id_handle.from_numpy(spk_id)
    phone_ids = frontend_dict['phone_ids']
    phones = phone_ids[0].numpy()
    phones_handle = am_predictor.get_input(0)
    phones_handle.from_numpy(phones)
    am_predictor.run()
    am_output_handle = am_predictor.get_output(0)
    am_output_data = am_output_handle.numpy()
    return am_output_data


def get_lite_voc_output(voc_predictor, input):
    mel_handle = voc_predictor.get_input(0)
    mel_handle.from_numpy(input)
    voc_predictor.run()
    voc_output_handle = voc_predictor.get_output(0)
    wav = voc_output_handle.numpy()
    return wav


def get_lite_am_sublayer_output(am_sublayer_predictor, input):
    input_handle = am_sublayer_predictor.get_input(0)
    input_handle.from_numpy(input)

    am_sublayer_predictor.run()
    am_sublayer_handle = am_sublayer_predictor.get_output(0)
    am_sublayer_output = am_sublayer_handle.numpy()
    return am_sublayer_output


def get_lite_streaming_am_output(input: str,
                                 am_encoder_infer_predictor,
                                 am_decoder_predictor,
                                 am_postnet_predictor,
                                 frontend,
                                 lang: str = 'zh',
                                 merge_sentences: bool = True):
    get_tone_ids = False
    frontend_dict = run_frontend(frontend=frontend,
                                 text=input,
                                 merge_sentences=merge_sentences,
                                 get_tone_ids=get_tone_ids,
                                 lang=lang)
    phone_ids = frontend_dict['phone_ids']
    phones = phone_ids[0].numpy()
    am_encoder_infer_output = get_lite_am_sublayer_output(
        am_encoder_infer_predictor, input=phones)
    am_decoder_output = get_lite_am_sublayer_output(
        am_decoder_predictor, input=am_encoder_infer_output)
    am_postnet_output = get_lite_am_sublayer_output(am_postnet_predictor,
                                                    input=np.transpose(
                                                        am_decoder_output,
                                                        (0, 2, 1)))
    am_output_data = am_decoder_output + np.transpose(am_postnet_output,
                                                      (0, 2, 1))
    normalized_mel = am_output_data[0]
    return normalized_mel
