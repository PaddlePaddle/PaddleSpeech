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
import argparse
import os
import sys
from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import paddle
import soundfile
from paddleaudio.backends import soundfile_load as load_audio
from paddleaudio.compliance.librosa import melspectrogram
from yacs.config import CfgNode

from ..executor import BaseExecutor
from ..log import logger
from ..utils import stats_wrapper
from paddlespeech.vector.io.batch import feature_normalize
from paddlespeech.vector.modules.sid_model import SpeakerIdetification


class VectorExecutor(BaseExecutor):
    def __init__(self):
        super().__init__('vector')
        self.parser = argparse.ArgumentParser(prog="paddlespeech.vector",
                                              add_help=True)

        self.parser.add_argument(
            "--model",
            type=str,
            default="ecapatdnn_voxceleb12",
            choices=[
                tag[:tag.index('-')]
                for tag in self.task_resource.pretrained_models.keys()
            ],
            help="Choose model type of vector task.")
        self.parser.add_argument("--task",
                                 type=str,
                                 default="spk",
                                 choices=["spk", "score"],
                                 help="task type in vector domain")
        self.parser.add_argument("--input",
                                 type=str,
                                 default=None,
                                 help="Audio file to extract embedding.")
        self.parser.add_argument(
            "--sample_rate",
            type=int,
            default=16000,
            choices=[16000],
            help="Choose the audio sample rate of the model. 8000 or 16000")
        self.parser.add_argument("--ckpt_path",
                                 type=str,
                                 default=None,
                                 help="Checkpoint file of model.")
        self.parser.add_argument('--yes',
                                 '-y',
                                 action="store_true",
                                 default=False,
                                 help='No additional parameters required. \
            Once set this parameter, it means accepting the request of the program by default, \
            which includes transforming the audio sample rate')
        self.parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Config of asr task. Use deault config when it is None.')
        self.parser.add_argument(
            "--device",
            type=str,
            default=paddle.get_device(),
            help="Choose device to execute model inference.")
        self.parser.add_argument('-d',
                                 '--job_dump_result',
                                 action='store_true',
                                 help='Save job result into file.')

        self.parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            help='Increase logger verbosity of current task.')

    def execute(self, argv: List[str]) -> bool:
        """Command line entry for vector model

        Args:
            argv (List[str]): command line args list

        Returns:
            bool: 
                False: some audio occurs error
                True: all audio process success
        """
        # stage 0: parse the args and get the required args
        parser_args = self.parser.parse_args(argv)
        model = parser_args.model
        sample_rate = parser_args.sample_rate
        config = parser_args.config
        ckpt_path = parser_args.ckpt_path
        force_yes = parser_args.yes
        device = parser_args.device

        # stage 1: configurate the verbose flag
        if not parser_args.verbose:
            self.disable_task_loggers()

        # stage 2: read the input data and store them as a list
        task_source = self.get_input_source(parser_args.input)
        logger.debug(f"task source: {task_source}")

        # stage 3: process the audio one by one
        # we do action according the task type
        task_result = OrderedDict()
        has_exceptions = False
        for id_, input_ in task_source.items():
            try:
                # extract the speaker audio embedding
                if parser_args.task == "spk":
                    logger.debug("do vector spk task")
                    res = self(audio_file=input_,
                               model=model,
                               sample_rate=sample_rate,
                               config=config,
                               ckpt_path=ckpt_path,
                               force_yes=force_yes,
                               device=device)
                    task_result[id_] = res
                elif parser_args.task == "score":
                    logger.debug("do vector score task")
                    logger.debug(f"input content {input_}")
                    if len(input_.split()) != 2:
                        logger.error(
                            f"vector score task input {input_} wav num is not two,"
                            "that is {len(input_.split())}")
                        sys.exit(-1)

                    # get the enroll and test embedding
                    enroll_audio, test_audio = input_.split()
                    logger.debug(
                        f"score task, enroll audio: {enroll_audio}, test audio: {test_audio}"
                    )
                    enroll_embedding = self(audio_file=enroll_audio,
                                            model=model,
                                            sample_rate=sample_rate,
                                            config=config,
                                            ckpt_path=ckpt_path,
                                            force_yes=force_yes,
                                            device=device)
                    test_embedding = self(audio_file=test_audio,
                                          model=model,
                                          sample_rate=sample_rate,
                                          config=config,
                                          ckpt_path=ckpt_path,
                                          force_yes=force_yes,
                                          device=device)

                    # get the score
                    res = self.get_embeddings_score(enroll_embedding,
                                                    test_embedding)
                    task_result[id_] = res
            except Exception as e:
                has_exceptions = True
                task_result[id_] = f'{e.__class__.__name__}: {e}'

        logger.debug("task result as follows: ")
        logger.debug(f"{task_result}")

        # stage 4: process the all the task results
        self.process_task_results(parser_args.input, task_result,
                                  parser_args.job_dump_result)

        # stage 5: return the exception flag
        #          if return False, somen audio process occurs error
        if has_exceptions:
            return False
        else:
            return True

    def _get_job_contents(
            self, job_input: os.PathLike) -> Dict[str, Union[str, os.PathLike]]:
        """
        Read a job input file and return its contents in a dictionary.
        Refactor from the Executor._get_job_contents

        Args:
            job_input (os.PathLike): The job input file.

        Returns:
            Dict[str, str]: Contents of job input.
        """
        job_contents = OrderedDict()
        with open(job_input) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                k = line.split(' ')[0]
                v = ' '.join(line.split(' ')[1:])
                job_contents[k] = v
        return job_contents

    def get_embeddings_score(self, enroll_embedding, test_embedding):
        """get the enroll embedding and test embedding score

        Args:
            enroll_embedding (numpy.array): shape: (emb_size), enroll audio embedding
            test_embedding (numpy.array): shape: (emb_size), test audio embedding

        Returns:
            score: the score between enroll embedding and test embedding
        """
        if not hasattr(self, "score_func"):
            self.score_func = paddle.nn.CosineSimilarity(axis=0)
            logger.debug("create the cosine score function ")

        score = self.score_func(paddle.to_tensor(enroll_embedding),
                                paddle.to_tensor(test_embedding))

        return score.item()

    @stats_wrapper
    def __call__(self,
                 audio_file: os.PathLike,
                 model: str = 'ecapatdnn_voxceleb12',
                 sample_rate: int = 16000,
                 config: os.PathLike = None,
                 ckpt_path: os.PathLike = None,
                 force_yes: bool = False,
                 device=paddle.get_device()):
        """Extract the audio embedding

        Args:
            audio_file (os.PathLike): audio path, 
                                      whose format must be wav and sample rate must be matched the model
            model (str, optional): mode type, which is been loaded from the pretrained model list. 
                                   Defaults to 'ecapatdnn-voxceleb12'.
            sample_rate (int, optional): model sample rate. Defaults to 16000.
            config (os.PathLike, optional): yaml config. Defaults to None.
            ckpt_path (os.PathLike, optional): pretrained model path. Defaults to None.
            device (optional): paddle running host device. Defaults to paddle.get_device().

        Returns:
            dict: return the audio embedding and the embedding shape
        """
        # stage 0: check the audio format
        audio_file = os.path.abspath(audio_file)
        if not self._check(audio_file, sample_rate, force_yes):
            sys.exit(-1)

        # stage 1: set the paddle runtime host device
        logger.debug(f"device type: {device}")
        paddle.device.set_device(device)

        # stage 2: read the specific pretrained model
        self._init_from_path(model, sample_rate, config, ckpt_path)

        # stage 3: preprocess the audio and get the audio feat
        self.preprocess(model, audio_file)

        # stage 4: infer the model and get the audio embedding
        self.infer(model)

        # stage 5: process the result and set them to output dict
        res = self.postprocess()

        return res

    def _init_from_path(self,
                        model_type: str = 'ecapatdnn_voxceleb12',
                        sample_rate: int = 16000,
                        cfg_path: Optional[os.PathLike] = None,
                        ckpt_path: Optional[os.PathLike] = None,
                        task=None):
        """Init the neural network from the model path

        Args:
            model_type (str, optional): model tag in the pretrained model list. 
                                        Defaults to 'ecapatdnn_voxceleb12'.
            sample_rate (int, optional): model sample rate. 
                                         Defaults to 16000.
            cfg_path (Optional[os.PathLike], optional): yaml config file path. 
                                                        Defaults to None.
            ckpt_path (Optional[os.PathLike], optional): the pretrained model path, which is stored in the disk. 
                                                         Defaults to None.
            task (str, optional): the model task type
        """
        # stage 0: avoid to init the mode again
        self.task = task
        if hasattr(self, "model"):
            logger.debug("Model has been initialized")
            return

        # stage 1: get the model and config path
        #          if we want init the network from the model stored in the disk,
        #          we must pass the config path and the ckpt model path
        if cfg_path is None or ckpt_path is None:
            # get the mode from pretrained list
            sample_rate_str = "16k" if sample_rate == 16000 else "8k"
            tag = model_type + "-" + sample_rate_str
            self.task_resource.set_task_model(tag, version=None)
            logger.debug(f"load the pretrained model: {tag}")
            # get the model from the pretrained list
            # we download the pretrained model and store it in the res_path
            self.res_path = self.task_resource.res_dir

            self.cfg_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['cfg_path'])
            self.ckpt_path = os.path.join(
                self.task_resource.res_dir,
                self.task_resource.res_dict['ckpt_path'] + '.pdparams')
        else:
            # get the model from disk
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path + ".pdparams")
            self.res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))

        logger.debug(f"start to read the ckpt from {self.ckpt_path}")
        logger.debug(f"read the config from {self.cfg_path}")
        logger.debug(f"get the res path {self.res_path}")

        # stage 2: read and config and init the model body
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)

        # stage 3: get the model name to instance the model network with dynamic_import
        logger.debug("start to dynamic import the model class")
        model_name = model_type[:model_type.rindex('_')]
        model_class = self.task_resource.get_model_class(model_name)
        logger.debug(f"model name {model_name}")
        model_conf = self.config.model
        backbone = model_class(**model_conf)
        model = SpeakerIdetification(backbone=backbone,
                                     num_class=self.config.num_speakers)
        self.model = model
        self.model.eval()

        # stage 4: load the model parameters
        logger.debug("start to set the model parameters to model")
        model_dict = paddle.load(self.ckpt_path)
        self.model.set_state_dict(model_dict)

        logger.debug("create the model instance success")

    @paddle.no_grad()
    def infer(self, model_type: str):
        """Infer the model to get the embedding

        Args:
            model_type (str): speaker verification model type
        """
        # stage 0: get the feat and length from _inputs
        feats = self._inputs["feats"]
        lengths = self._inputs["lengths"]
        logger.debug("start to do backbone network model forward")
        logger.debug(
            f"feats shape:{feats.shape}, lengths shape: {lengths.shape}")

        # stage 1: get the audio embedding
        # embedding from (1, emb_size, 1) -> (emb_size)
        embedding = self.model.backbone(feats, lengths).squeeze().numpy()
        logger.debug(f"embedding size: {embedding.shape}")

        # stage 2: put the embedding and dim info to _outputs property
        #          the embedding type is numpy.array
        self._outputs["embedding"] = embedding

    def postprocess(self) -> Union[str, os.PathLike]:
        """Return the audio embedding info

        Returns:
            Union[str, os.PathLike]: audio embedding info
        """
        embedding = self._outputs["embedding"]
        return embedding

    def preprocess(self, model_type: str, input_file: Union[str, os.PathLike]):
        """Extract the audio feat

        Args:
            model_type (str): speaker verification model type
            input_file (Union[str, os.PathLike]): audio file path
        """
        audio_file = input_file
        if isinstance(audio_file, (str, os.PathLike)):
            logger.debug(f"Preprocess audio file: {audio_file}")

        # stage 1: load the audio sample points
        #    Note: this process must match the training process
        waveform, sr = load_audio(audio_file)
        logger.debug(
            f"load the audio sample points, shape is: {waveform.shape}")

        # stage 2: get the audio feat
        # Note: Now we only support fbank feature
        try:
            feat = melspectrogram(x=waveform,
                                  sr=self.config.sr,
                                  n_mels=self.config.n_mels,
                                  window_size=self.config.window_size,
                                  hop_length=self.config.hop_size)
            logger.debug(f"extract the audio feat, shape is: {feat.shape}")
        except Exception as e:
            logger.debug(f"feat occurs exception {e}")
            sys.exit(-1)

        feat = paddle.to_tensor(feat).unsqueeze(0)
        # in inference period, the lengths is all one without padding
        lengths = paddle.ones([1])

        # stage 3: we do feature normalize,
        #          Now we assume that the feat must do normalize
        feat = feature_normalize(feat, mean_norm=True, std_norm=False)

        # stage 4: store the feat and length in the _inputs,
        #          which will be used in other function
        logger.debug(f"feats shape: {feat.shape}")
        self._inputs["feats"] = feat
        self._inputs["lengths"] = lengths

        logger.debug("audio extract the feat success")

    def _check(self,
               audio_file: str,
               sample_rate: int,
               force_yes: bool = False):
        """Check if the model sample match the audio sample rate 

        Args:
            audio_file (str): audio file path, which will be extracted the embedding
            sample_rate (int): the desired model sample rate 

        Returns:
            bool: return if the audio sample rate matches the model sample rate
        """
        self.sample_rate = sample_rate
        if self.sample_rate != 16000 and self.sample_rate != 8000:
            logger.error(
                "invalid sample rate, please input --sr 8000 or --sr 16000")
            logger.error(
                f"The model sample rate: {self.sample_rate}, the external sample rate is: {sample_rate}"
            )
            return False

        if isinstance(audio_file, (str, os.PathLike)):
            if not os.path.isfile(audio_file):
                logger.error("Please input the right audio file path")
                return False

        logger.debug("checking the aduio file format......")
        try:
            audio, audio_sample_rate = soundfile.read(audio_file,
                                                      dtype="float32",
                                                      always_2d=True)
        except Exception as e:
            logger.exception(e)
            logger.error(
                "can not open the audio file, please check the audio file format is 'wav'. \n \
                 you can try to use sox to change the file format.\n \
                 For example: \n \
                 sample rate: 16k \n \
                 sox input_audio.xx --rate 16k --bits 16 --channels 1 output_audio.wav \n \
                 sample rate: 8k \n \
                 sox input_audio.xx --rate 8k --bits 16 --channels 1 output_audio.wav \n \
                 ")
            return False

        logger.debug(f"The sample rate is {audio_sample_rate}")

        if audio_sample_rate != self.sample_rate:
            logger.debug("The sample rate of the input file is not {}.\n \
                            The program will resample the wav file to {}.\n \
                            If the result does not meet your expectations，\n \
                            Please input the 16k 16 bit 1 channel wav file. \
                        ".format(self.sample_rate, self.sample_rate))
            if force_yes is False:
                while (True):
                    logger.debug(
                        "Whether to change the sample rate and the channel. Y: change the sample. N: exit the prgream."
                    )
                    content = input("Input(Y/N):")
                    if content.strip() == "Y" or content.strip(
                    ) == "y" or content.strip() == "yes" or content.strip(
                    ) == "Yes":
                        logger.debug(
                            "change the sampele rate, channel to 16k and 1 channel"
                        )
                        break
                    elif content.strip() == "N" or content.strip(
                    ) == "n" or content.strip() == "no" or content.strip(
                    ) == "No":
                        logger.debug("Exit the program")
                        return False
                    else:
                        logger.warning("Not regular input, please input again")
            self.change_format = True
        else:
            logger.debug("The audio file format is right")
            self.change_format = False

        return True
