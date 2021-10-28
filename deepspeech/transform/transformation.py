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
"""Transformation module."""
import copy
import io
import logging
from collections import OrderedDict
from collections.abc import Sequence
from inspect import signature

import yaml

from deepspeech.utils.dynamic_import import dynamic_import

# TODO(karita): inherit TransformInterface
# TODO(karita): register cmd arguments in asr_train.py
import_alias = dict(
    identity="deepspeech.transform.transform_interface:Identity",
    time_warp="deepspeech.transform.spec_augment:TimeWarp",
    time_mask="deepspeech.transform.spec_augment:TimeMask",
    freq_mask="deepspeech.transform.spec_augment:FreqMask",
    spec_augment="deepspeech.transform.spec_augment:SpecAugment",
    speed_perturbation="deepspeech.transform.perturb:SpeedPerturbation",
    volume_perturbation="deepspeech.transform.perturb:VolumePerturbation",
    noise_injection="deepspeech.transform.perturb:NoiseInjection",
    bandpass_perturbation="deepspeech.transform.perturb:BandpassPerturbation",
    rir_convolve="deepspeech.transform.perturb:RIRConvolve",
    delta="deepspeech.transform.add_deltas:AddDeltas",
    cmvn="deepspeech.transform.cmvn:CMVN",
    utterance_cmvn="deepspeech.transform.cmvn:UtteranceCMVN",
    fbank="deepspeech.transform.spectrogram:LogMelSpectrogram",
    spectrogram="deepspeech.transform.spectrogram:Spectrogram",
    stft="deepspeech.transform.spectrogram:Stft",
    istft="deepspeech.transform.spectrogram:IStft",
    stft2fbank="deepspeech.transform.spectrogram:Stft2LogMelSpectrogram",
    wpe="deepspeech.transform.wpe:WPE",
    channel_selector="deepspeech.transform.channel_selector:ChannelSelector", )


class Transformation():
    """Apply some functions to the mini-batch

    Examples:
        >>> kwargs = {"process": [{"type": "fbank",
        ...                        "n_mels": 80,
        ...                        "fs": 16000},
        ...                       {"type": "cmvn",
        ...                        "stats": "data/train/cmvn.ark",
        ...                        "norm_vars": True},
        ...                       {"type": "delta", "window": 2, "order": 2}]}
        >>> transform = Transformation(kwargs)
        >>> bs = 10
        >>> xs = [np.random.randn(100, 80).astype(np.float32)
        ...       for _ in range(bs)]
        >>> xs = transform(xs)
    """

    def __init__(self, conffile=None):
        if conffile is not None:
            if isinstance(conffile, dict):
                self.conf = copy.deepcopy(conffile)
            else:
                with io.open(conffile, encoding="utf-8") as f:
                    self.conf = yaml.safe_load(f)
                    assert isinstance(self.conf, dict), type(self.conf)
        else:
            self.conf = {"mode": "sequential", "process": []}

        self.functions = OrderedDict()
        if self.conf.get("mode", "sequential") == "sequential":
            for idx, process in enumerate(self.conf["process"]):
                assert isinstance(process, dict), type(process)
                opts = dict(process)
                process_type = opts.pop("type")
                class_obj = dynamic_import(process_type, import_alias)
                # TODO(karita): assert issubclass(class_obj, TransformInterface)
                try:
                    self.functions[idx] = class_obj(**opts)
                except TypeError:
                    try:
                        signa = signature(class_obj)
                    except ValueError:
                        # Some function, e.g. built-in function, are failed
                        pass
                    else:
                        logging.error("Expected signature: {}({})".format(
                            class_obj.__name__, signa))
                    raise
        else:
            raise NotImplementedError(
                "Not supporting mode={}".format(self.conf["mode"]))

    def __repr__(self):
        rep = "\n" + "\n".join("    {}: {}".format(k, v)
                               for k, v in self.functions.items())
        return "{}({})".format(self.__class__.__name__, rep)

    def __call__(self, xs, uttid_list=None, **kwargs):
        """Return new mini-batch

        :param Union[Sequence[np.ndarray], np.ndarray] xs:
        :param Union[Sequence[str], str] uttid_list:
        :return: batch:
        :rtype: List[np.ndarray]
        """
        if not isinstance(xs, Sequence):
            is_batch = False
            xs = [xs]
        else:
            is_batch = True

        if isinstance(uttid_list, str):
            uttid_list = [uttid_list for _ in range(len(xs))]

        if self.conf.get("mode", "sequential") == "sequential":
            for idx in range(len(self.conf["process"])):
                func = self.functions[idx]
                # TODO(karita): use TrainingTrans and UttTrans to check __call__ args
                # Derive only the args which the func has
                try:
                    param = signature(func).parameters
                except ValueError:
                    # Some function, e.g. built-in function, are failed
                    param = {}
                _kwargs = {k: v for k, v in kwargs.items() if k in param}
                try:
                    if uttid_list is not None and "uttid" in param:
                        xs = [
                            func(x, u, **_kwargs)
                            for x, u in zip(xs, uttid_list)
                        ]
                    else:
                        xs = [func(x, **_kwargs) for x in xs]
                except Exception:
                    logging.fatal("Catch a exception from {}th func: {}".format(
                        idx, func))
                    raise
        else:
            raise NotImplementedError(
                "Not supporting mode={}".format(self.conf["mode"]))

        if is_batch:
            return xs
        else:
            return xs[0]
