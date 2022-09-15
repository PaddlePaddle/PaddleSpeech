"""
G2p Voice Clone
"""

import os
from paddlespeech.t2s.modules.normalizer import ZScore
import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.utils.dynamic_import import dynamic_import
from paddlespeech.cli.vector import VectorExecutor



model_alias = {
    # acoustic model
    "fastspeech2":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2",
    "fastspeech2_inference":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2Inference",
    # voc
    "pwgan":
    "paddlespeech.t2s.models.parallel_wavegan:PWGGenerator",
    "pwgan_inference":
    "paddlespeech.t2s.models.parallel_wavegan:PWGInference",
}


# 设置预训练模型的路径和其他变量
# am


class VoiceCloneTDNN():
    def __init__(self):
        
        self.am = "fastspeech2_aishell3"
        self.am_config = "source/model/fastspeech2_aishell3_ckpt_vc2_1.2.0/default.yaml"
        self.am_ckpt = "source/model/fastspeech2_aishell3_ckpt_vc2_1.2.0/snapshot_iter_96400.pdz"
        self.am_stat = "source/model/fastspeech2_aishell3_ckpt_vc2_1.2.0/speech_stats.npy"
        self.phones_dict = "source/model/fastspeech2_aishell3_ckpt_vc2_1.2.0/phone_id_map.txt"
        # voc
        self.voc = "pwgan_aishell3"
        self.voc_config = "source/model/pwg_aishell3_ckpt_0.5/default.yaml"
        self.voc_ckpt = "source/model/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz"
        self.voc_stat = "source/model/pwg_aishell3_ckpt_0.5/feats_stats.npy"
        
        with open(self.am_config) as f:
            self.am_config = CfgNode(yaml.safe_load(f))
        with open(self.voc_config) as f:
            self.voc_config = CfgNode(yaml.safe_load(f))
        self.vec_executor = VectorExecutor()
        
        
        with open(self.phones_dict, "r") as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)
        
        self.frontend = Frontend(phone_vocab_path=self.phones_dict)
        
        # am
        am_name = "fastspeech2"
        am_class = dynamic_import(am_name, model_alias)
        print(self.am_config.n_mels)
        self.am = am_class(
            idim=vocab_size, odim=self.am_config.n_mels, spk_num=None, **self.am_config["model"])
        self.am_inference_class = dynamic_import(am_name + '_inference', model_alias)
        self.am.set_state_dict(paddle.load(self.am_ckpt)["main_params"])
        self.am.eval()
        
        am_mu, am_std = np.load(self.am_stat)
        am_mu = paddle.to_tensor(am_mu)
        am_std = paddle.to_tensor(am_std)
        self.am_normalizer = ZScore(am_mu, am_std)
        self.am_inference = self.am_inference_class(self.am_normalizer, self.am)
        self.am_inference.eval()
        
        # voc
        voc_name = "pwgan"
        voc_class = dynamic_import(voc_name, model_alias)
        voc_inference_class = dynamic_import(voc_name + '_inference', model_alias)
        self.voc = voc_class(**self.voc_config["generator_params"])
        self.voc.set_state_dict(paddle.load(self.voc_ckpt)["generator_params"])
        self.voc.remove_weight_norm()
        self.voc.eval()
        voc_mu, voc_std = np.load(self.voc_stat)
        voc_mu = paddle.to_tensor(voc_mu)
        voc_std = paddle.to_tensor(voc_std)
        voc_normalizer = ZScore(voc_mu, voc_std)
        self.voc_inference = voc_inference_class(voc_normalizer, self.voc)
        self.voc_inference.eval()
        
    def vc(self, text, input_wav, out_wav):
        input_ids = self.frontend.get_input_ids(text, merge_sentences=True)
        phone_ids = input_ids["phone_ids"][0]
        spk_emb = self.vec_executor(audio_file=input_wav, force_yes=True)
        spk_emb = paddle.to_tensor(spk_emb)

        with paddle.no_grad():
            wav = self.voc_inference(self.am_inference(phone_ids, spk_emb=spk_emb))
        sf.write(out_wav, wav.numpy(), samplerate=self.am_config.fs)
        return True

    
if __name__ == '__main__':
    voiceclone =VoiceCloneTDNN()
    text = "测试一下你的合成效果"
    input_wav = os.path.realpath("source/wav/test/009901.wav")
    out_wav = os.path.realpath("source/wav/test/9901_clone.wav")
    voiceclone.vc(text, input_wav, out_wav)