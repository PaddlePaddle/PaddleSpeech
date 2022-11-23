import os
import shutil

from .util import get_ngpu
from .util import MAIN_ROOT
from .util import run_cmd


class VoiceCloneTDNN():
    def __init__(self):
        # Path 到指定路径上
        self.BIN_DIR = os.path.join(MAIN_ROOT, "paddlespeech/t2s/exps")

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

    def vc(self, text, input_wav, out_wav):
        # input wav 需要形成临时单独文件夹
        _, full_file_name = os.path.split(input_wav)
        ref_audio_dir = os.path.realpath("tmp_dir/tdnn")
        if os.path.exists(ref_audio_dir):
            shutil.rmtree(ref_audio_dir)
        os.makedirs(ref_audio_dir, exist_ok=True)
        shutil.copy(input_wav, ref_audio_dir)

        output_dir = os.path.dirname(out_wav)
        ngpu = get_ngpu()

        cmd = f"""
            python3 {self.BIN_DIR}/voice_cloning.py \
                    --am={self.am} \
                    --am_config={self.am_config} \
                    --am_ckpt={self.am_ckpt} \
                    --am_stat={self.am_stat} \
                    --voc={self.voc} \
                    --voc_config={self.voc_config} \
                    --voc_ckpt={self.voc_ckpt} \
                    --voc_stat={self.voc_stat} \
                    --text="{text}" \
                    --input-dir={ref_audio_dir} \
                    --output-dir={output_dir} \
                    --phones-dict={self.phones_dict} \
                    --use_ecapa=True \
                    --ngpu={ngpu}
        """

        output_name = os.path.join(output_dir, full_file_name)
        return run_cmd(cmd, output_name=output_name)
