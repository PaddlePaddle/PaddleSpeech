import os

from .util import MAIN_ROOT
from .util import run_cmd


def find_max_ckpt(model_path):
    max_ckpt = 0
    for filename in os.listdir(model_path):
        if filename.endswith('.pdz'):
            files = filename[:-4]
            a1, a2, it = files.split("_")
            if int(it) > max_ckpt:
                max_ckpt = int(it)
    return max_ckpt


class FineTune:
    def __init__(self):
        self.now_file_path = os.path.dirname(__file__)
        self.PYTHONPATH = os.path.join(MAIN_ROOT,
                                       "examples/other/tts_finetune/tts3")
        self.BIN_DIR = os.path.join(MAIN_ROOT,
                                    "paddlespeech/t2s/exps/fastspeech2")
        self.pretrained_model_dir = os.path.realpath(
            "source/model/fastspeech2_aishell3_ckpt_1.1.0")
        self.voc_model_dir = os.path.realpath(
            "source/model/hifigan_aishell3_ckpt_0.2.0")
        self.finetune_config = os.path.join("conf/tts3_finetune.yaml")

    def finetune(self, input_dir, exp_dir='temp', epoch=100):
        """
        use cmd follow examples/other/tts_finetune/tts3/run.sh
        """
        newdir_name = "newdir"
        new_dir = os.path.join(input_dir, newdir_name)
        mfa_dir = os.path.join(exp_dir, 'mfa_result')
        dump_dir = os.path.join(exp_dir, 'dump')
        output_dir = os.path.join(exp_dir, 'exp')
        lang = "zh"
        ngpu = 1

        cmd = f"""
            # check oov
            python3 {self.PYTHONPATH}/local/check_oov.py \
                --input_dir={input_dir} \
                --pretrained_model_dir={self.pretrained_model_dir} \
                --newdir_name={newdir_name} \
                --lang={lang}
            
            # get mfa result
            python3 {self.PYTHONPATH}/local/get_mfa_result.py \
                --input_dir={new_dir} \
                --mfa_dir={mfa_dir} \
                --lang={lang}
            
            # generate durations.txt
            python3 {self.PYTHONPATH}/local/generate_duration.py \
                --mfa_dir={mfa_dir} 
            
            # extract feature
            python3 {self.PYTHONPATH}/local/extract_feature.py \
                --duration_file="./durations.txt" \
                --input_dir={new_dir} \
                --dump_dir={dump_dir} \
                --pretrained_model_dir={self.pretrained_model_dir}
            
            # create finetune env
            python3 {self.PYTHONPATH}/local/prepare_env.py \
                --pretrained_model_dir={self.pretrained_model_dir} \
                --output_dir={output_dir}
            
            # finetune
            python3 {self.PYTHONPATH}/local/finetune.py \
                --pretrained_model_dir={self.pretrained_model_dir} \
                --dump_dir={dump_dir} \
                --output_dir={output_dir} \
                --ngpu={ngpu} \
                --epoch=100 \
                --finetune_config={self.finetune_config}
        """

        print(cmd)

        return run_cmd(cmd, exp_dir)

    def synthesize(self, text, wav_name, out_wav_dir, exp_dir='temp'):

        voc = "hifigan_aishell3"
        dump_dir = os.path.join(exp_dir, 'dump')
        output_dir = os.path.join(exp_dir, 'exp')
        text_path = os.path.join(exp_dir, 'sentences.txt')
        lang = "zh"
        ngpu = 1

        model_path = f"{output_dir}/checkpoints"
        ckpt = find_max_ckpt(model_path)

        # 生成对应的语句
        with open(text_path, "w", encoding='utf8') as f:
            f.write(wav_name + " " + text)

        cmd = f"""
            FLAGS_allocator_strategy=naive_best_fit \
            FLAGS_fraction_of_gpu_memory_to_use=0.01 \
            python3 {self.BIN_DIR}/../synthesize_e2e.py \
                --am=fastspeech2_aishell3 \
                --am_config={self.pretrained_model_dir}/default.yaml \
                --am_ckpt={output_dir}/checkpoints/snapshot_iter_{ckpt}.pdz \
                --am_stat={self.pretrained_model_dir}/speech_stats.npy \
                --voc={voc} \
                --voc_config={self.voc_model_dir}/default.yaml \
                --voc_ckpt={self.voc_model_dir}/snapshot_iter_2500000.pdz \
                --voc_stat={self.voc_model_dir}/feats_stats.npy \
                --lang={lang} \
                --text={text_path} \
                --output_dir={out_wav_dir} \
                --phones_dict={dump_dir}/phone_id_map.txt \
                --speaker_dict={dump_dir}/speaker_id_map.txt \
                --spk_id=0 
        """

        out_path = os.path.join(out_wav_dir, f"{wav_name}.wav")

        return run_cmd(cmd, out_path)
