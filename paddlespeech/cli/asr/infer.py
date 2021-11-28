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
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import paddle
import soundfile
import wget

from paddlespeech.s2t.exps.u2.config import get_cfg_defaults
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig

logger = Log(__name__).getlog()

model_alias = {
    "ds2_offline": "paddlespeech.s2t.models.ds2:DeepSpeech2Model",
    "ds2_online": "paddlespeech.s2t.models.ds2_online:DeepSpeech2ModelOnline",
    "conformer": "paddlespeech.s2t.models.u2:U2Model",
    "transformer": "paddlespeech.s2t.models.u2:U2Model",
    "wenetspeech": "paddlespeech.s2t.models.u2:U2Model",
}

pretrain_model_alias = {
    "ds2_online_zn": [
        "https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/aishell_ds2_online_cer8.00_release.tar.gz",
        "", ""
    ],
    "ds2_offline_zn": [
        "https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/ds2.model.tar.gz",
        "", ""
    ],
    "transformer_zn": [
        "https://paddlespeech.bj.bcebos.com/s2t/aishell/asr1/transformer.model.tar.gz",
        "", ""
    ],
    "conformer_zn": [
        "https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/conformer.model.tar.gz",
        "", ""
    ],
    "wenetspeech_zn": [
        "https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/conformer.model.tar.gz",
        "conf/conformer.yaml", "exp/conformer/checkpoints/wenetspeech"
    ],
}


class BaseInfer():
    def __init__(self, config, args):
        self.args = args
        self.config = config
        self.audio_file = args.audio_file
        paddle.set_device('gpu' if self.args.ngpu > 0 else 'cpu')

        self.sr = config.collator.target_sample_rate

        self.text_feature = TextFeaturizer(
            unit_type=config.collator.unit_type,
            vocab_filepath=config.collator.vocab_filepath,
            spm_model_prefix=config.collator.spm_model_prefix)

        # Get the object for feature extraction
        if args.model_name == "ds2_online" or args.model_name == "ds2_offline":
            self.collate_fn_test = SpeechCollator.from_config(config)
        elif args.model_name == "conformer" or args.model_name == "transformer" or args.model_name == "wenetspeech":
            self.preprocess_conf = config.collator.augmentation_config
            self.preprocess_args = {"train": False}
            self.preprocessing = Transformation(self.preprocess_conf)
        else:
            raise Exception("wrong type")

        # model
        model_conf = config.model
        logger.info(model_conf)
        with UpdateConfig(model_conf):
            if args.model_name == "ds2_online" or args.model_name == "ds2_offline":
                model_conf.feat_size = self.collate_fn_test.feature_size
                model_conf.dict_size = self.text_feature.vocab_size
            elif args.model_name == "conformer" or args.model_name == "transformer" or args.model_name == "wenetspeech":
                model_conf.input_dim = config.collator.feat_dim
                model_conf.output_dim = self.text_feature.vocab_size
            else:
                raise Exception("wrong type")
        model_class = dynamic_import(args.model_name, model_alias)
        model = model_class.from_config(model_conf)
        self.model = model
        self.model.eval()

        # load model
        params_path = self.args.checkpoint_path + ".pdparams"
        model_dict = paddle.load(params_path)
        self.model.set_state_dict(model_dict)

    def run(self):
        check(args.audio_file)

        with paddle.no_grad():
            # read the audio file
            if args.model_name == "ds2_online" or args.model_name == "ds2_offline":
                collate_fn_test = self.collate_fn_test
                audio, _ = collate_fn_test.process_utterance(
                    audio_file=self.audio_file, transcript=" ")
                audio_len = audio.shape[0]
                audio = paddle.to_tensor(audio, dtype='float32')
                audio_len = paddle.to_tensor(audio_len)
                audio = paddle.unsqueeze(audio, axis=0)
                vocab_list = collate_fn_test.vocab_list

            elif args.model_name == "conformer" or args.model_name == "transformer" or args.model_name == "wenetspeech":
                audio, sample_rate = soundfile.read(
                    self.audio_file, dtype="int16", always_2d=True)
                if sample_rate != self.sr:
                    logger.error(
                        f"sample rate error: {sample_rate}, need {self.sr} ")
                    sys.exit(-1)

                audio = audio[:, 0]
                logger.info(f"audio shape: {audio.shape}")

                # fbank
                feat = self.preprocessing(audio, **self.preprocess_args)
                logger.info(f"feat shape: {feat.shape}")

                ilen = paddle.to_tensor(feat.shape[0])
                xs = paddle.to_tensor(feat, dtype='float32').unsqueeze(axis=0)

            else:
                raise Exception("wrong type")

            cfg = self.config.decoding

            if args.model_name == "ds2_online" or args.model_name == "ds2_offline":
                result_transcripts = self.model.decode(
                    audio,
                    audio_len,
                    vocab_list,
                    decoding_method=cfg.decoding_method,
                    lang_model_path=cfg.lang_model_path,
                    beam_alpha=cfg.alpha,
                    beam_beta=cfg.beta,
                    beam_size=cfg.beam_size,
                    cutoff_prob=cfg.cutoff_prob,
                    cutoff_top_n=cfg.cutoff_top_n,
                    num_processes=cfg.num_proc_bsearch)
                result_transcripts = result_transcripts[0]

            elif args.model_name == "conformer" or args.model_name == "transformer" or args.model_name == "wenetspeech":
                result_transcripts = self.model.decode(
                    xs,
                    ilen,
                    text_feature=self.text_feature,
                    decoding_method=cfg.decoding_method,
                    lang_model_path=cfg.lang_model_path,
                    beam_alpha=cfg.alpha,
                    beam_beta=cfg.beta,
                    beam_size=cfg.beam_size,
                    cutoff_prob=cfg.cutoff_prob,
                    cutoff_top_n=cfg.cutoff_top_n,
                    num_processes=cfg.num_proc_bsearch,
                    ctc_weight=cfg.ctc_weight,
                    decoding_chunk_size=cfg.decoding_chunk_size,
                    num_decoding_left_chunks=cfg.num_decoding_left_chunks,
                    simulate_streaming=cfg.simulate_streaming)
                result_transcripts = result_transcripts[0][0]
            else:
                raise Exception("invalid model name")

            rsl = result_transcripts
            utt = Path(self.audio_file).name
            logger.info(f"hyp: {utt} {result_transcripts}")
            return rsl


def check(audio_file):
    if not os.path.isfile(audio_file):
        logger.error("Please input the right audio file path")
        sys.exit(-1)

    logger.info("checking the audio file format......")
    try:
        sig, sample_rate = soundfile.read(audio_file)
    except Exception as e:
        logger.error(str(e))
        logger.error(
            "can not open the wav file, please check the audio file format")
        sys.exit(-1)
    logger.info("The sample rate is %d" % sample_rate)
    assert (sample_rate == 16000)
    logger.info("The audio file format is right")


def get_pretrain_model(args):
    pretrain_model_tag = args.model_name + "_" + args.language
    if pretrain_model_tag not in pretrain_model_alias.keys():
        logger.error(
            "please input the right model name and language type:\n model_name list: ['ds2_online', 'ds2_offline', 'conformer', 'transformer', 'wenetspeech']\n language list:['zn', 'en']"
        )
        logger.error("current support:" + str(
            list(pretrain_model_alias.keys())))
        sys.exit(-1)
    url = pretrain_model_alias[pretrain_model_tag][0]
    conf_path = pretrain_model_alias[pretrain_model_tag][1]
    ckpt_prefix = pretrain_model_alias[pretrain_model_tag][2]
    tmpdir = tempfile.gettempdir()
    target_name = wget.filename_from_url(url)
    logger.info("start downloading the model......")
    if not os.path.exists(os.path.join(tmpdir, pretrain_model_tag)):
        os.makedirs(os.path.join(tmpdir, pretrain_model_tag))

    file_path = wget.download(
        url, out=os.path.join(tmpdir, pretrain_model_tag, target_name))
    logger.info("download the file at:{}".format(file_path))
    tar = tarfile.open(file_path)
    tar.extractall(os.path.join(tmpdir, pretrain_model_tag))
    tar.close()
    # set the model config and the checkpoint path
    args.config = os.path.join(tmpdir, pretrain_model_tag, conf_path)
    args.checkpoint_path = os.path.join(tmpdir, pretrain_model_tag, ckpt_prefix)


def main(config, args):
    BaseInfer(config, args).run()


if __name__ == "__main__":
    parser = default_argument_parser()
    # save asr result to
    parser.add_argument(
        "--model_name",
        type=str,
        help="model_name list: ['ds2_online', 'ds2_offline', 'conformer', 'transformer', 'wenetspeech']"
    )
    parser.add_argument("--language", type=str, default="zn", help="zn or en")
    parser.add_argument(
        "--result_file", type=str, help="path of save the asr result")
    parser.add_argument(
        "--audio_file", type=str, help="path of the input audio file")
    parser.add_argument(
        "--use-default-model",
        action="store_true",
        help="use default pretrained model")
    args = parser.parse_args()
    if args.use_default_model is True:
        get_pretrain_model(args)

    config = get_cfg_defaults()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    main(config, args)
