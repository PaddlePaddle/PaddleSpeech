import os

from .util import run_cmd


class SAT:
    def __init__(self):
        # pretrain model path
        self.zh_pretrain_model_path = os.path.realpath(
            "source/model/erniesat_aishell3_ckpt_1.2.0")
        self.en_pretrain_model_path = os.path.realpath(
            "source/model/erniesat_vctk_ckpt_1.2.0")
        self.cross_pretrain_model_path = os.path.realpath(
            "source/model/erniesat_aishell3_vctk_ckpt_1.2.0")

        self.zh_voc_model_path = os.path.realpath(
            "source/model/hifigan_aishell3_ckpt_0.2.0")
        self.eb_voc_model_path = os.path.realpath(
            "source/model/hifigan_vctk_ckpt_0.2.0")
        self.cross_voc_model_path = os.path.realpath(
            "source/model/hifigan_aishell3_ckpt_0.2.0")

        self.now_file_path = os.path.dirname(__file__)
        self.BIN_DIR = os.path.realpath(
            os.path.join(self.now_file_path,
                         "../../../../paddlespeech/t2s/exps/ernie_sat"))

    def zh_synthesize_edit(self,
                           old_str: str,
                           new_str: str,
                           input_name: os.PathLike,
                           output_name: os.PathLike,
                           task_name: str="synthesize",
                           erniesat_ckpt_name: str="snapshot_iter_289500.pdz"):

        if task_name not in ['synthesize', 'edit']:
            print("task name only in ['edit', 'synthesize']")
            return None

        # 运行时的 PYTHONPATH
        PYTHONPATH = os.path.realpath(
            os.path.join(self.now_file_path,
                         "../../../../examples/aishell3/ernie_sat"))

        # 推理文件配置
        config_path = os.path.join(self.zh_pretrain_model_path, "default.yaml")
        phones_dict = os.path.join(self.zh_pretrain_model_path,
                                   "phone_id_map.txt")
        erniesat_ckpt = os.path.join(self.zh_pretrain_model_path,
                                     erniesat_ckpt_name)
        erniesat_stat = os.path.join(self.zh_pretrain_model_path,
                                     "speech_stats.npy")

        voc = "hifigan_aishell3"
        voc_config = os.path.join(self.zh_voc_model_path, "default.yaml")
        voc_ckpt = os.path.join(self.zh_voc_model_path,
                                "snapshot_iter_2500000.pdz")
        voc_stat = os.path.join(self.zh_voc_model_path, "feats_stats.npy")

        cmd = self.get_cmd(
            task_name,
            input_name,
            old_str,
            new_str,
            config_path,
            phones_dict,
            erniesat_ckpt,
            erniesat_stat,
            voc,
            voc_config,
            voc_ckpt,
            voc_stat,
            output_name,
            source_lang="zh",
            target_lang="zh")

        return run_cmd(cmd, output_name)

    def crossclone(self,
                   old_str: str,
                   new_str: str,
                   input_name: os.PathLike,
                   output_name: os.PathLike,
                   source_lang: str,
                   target_lang: str,
                   erniesat_ckpt_name: str="snapshot_iter_489000.pdz"):
        PYTHONPATH = os.path.realpath(
            os.path.join(self.now_file_path,
                         "../../../../examples/aishell3_vctk/ernie_sat"))
        # 推理文件配置
        config_path = os.path.join(self.cross_pretrain_model_path,
                                   "default.yaml")
        phones_dict = os.path.join(self.cross_pretrain_model_path,
                                   "phone_id_map.txt")
        erniesat_ckpt = os.path.join(self.cross_pretrain_model_path,
                                     erniesat_ckpt_name)
        erniesat_stat = os.path.join(self.cross_pretrain_model_path,
                                     "speech_stats.npy")

        voc = "hifigan_aishell3"
        voc_config = os.path.join(self.cross_voc_model_path, "default.yaml")
        voc_ckpt = os.path.join(self.cross_voc_model_path,
                                "snapshot_iter_2500000.pdz")
        voc_stat = os.path.join(self.cross_voc_model_path, "feats_stats.npy")
        task_name = "synthesize"
        cmd = self.get_cmd(task_name, input_name, old_str, new_str, config_path,
                           phones_dict, erniesat_ckpt, erniesat_stat, voc,
                           voc_config, voc_ckpt, voc_stat, output_name,
                           source_lang, target_lang)

        return run_cmd(cmd, output_name)

    def en_synthesize_edit(self,
                           old_str: str,
                           new_str: str,
                           input_name: os.PathLike,
                           output_name: os.PathLike,
                           task_name: str="synthesize",
                           erniesat_ckpt_name: str="snapshot_iter_199500.pdz"):
        PYTHONPATH = os.path.realpath(
            os.path.join(self.now_file_path,
                         "../../../../examples/vctk/ernie_sat"))

        # 推理文件配置
        config_path = os.path.join(self.en_pretrain_model_path, "default.yaml")
        phones_dict = os.path.join(self.en_pretrain_model_path,
                                   "phone_id_map.txt")
        erniesat_ckpt = os.path.join(self.en_pretrain_model_path,
                                     erniesat_ckpt_name)
        erniesat_stat = os.path.join(self.en_pretrain_model_path,
                                     "speech_stats.npy")

        voc = "hifigan_aishell3"
        voc_config = os.path.join(self.zh_voc_model_path, "default.yaml")
        voc_ckpt = os.path.join(self.zh_voc_model_path,
                                "snapshot_iter_2500000.pdz")
        voc_stat = os.path.join(self.zh_voc_model_path, "feats_stats.npy")

        cmd = self.get_cmd(
            task_name,
            input_name,
            old_str,
            new_str,
            config_path,
            phones_dict,
            erniesat_ckpt,
            erniesat_stat,
            voc,
            voc_config,
            voc_ckpt,
            voc_stat,
            output_name,
            source_lang="en",
            target_lang="en")

        return run_cmd(cmd, output_name)

    def get_cmd(self, task_name, input_name, old_str, new_str, config_path,
                phones_dict, erniesat_ckpt, erniesat_stat, voc, voc_config,
                voc_ckpt, voc_stat, output_name, source_lang, target_lang):
        cmd = f"""
            FLAGS_allocator_strategy=naive_best_fit \
            FLAGS_fraction_of_gpu_memory_to_use=0.01 \
            python3 {self.BIN_DIR}/synthesize_e2e.py \
                --task_name={task_name} \
                --wav_path={input_name} \
                --old_str='{old_str}' \
                --new_str='{new_str}' \
                --source_lang={source_lang} \
                --target_lang={target_lang} \
                --erniesat_config={config_path} \
                --phones_dict={phones_dict} \
                --erniesat_ckpt={erniesat_ckpt} \
                --erniesat_stat={erniesat_stat} \
                --voc={voc} \
                --voc_config={voc_config} \
                --voc_ckpt={voc_ckpt} \
                --voc_stat={voc_stat} \
                --output_name={output_name}
        """

        return cmd
