from .ernie_sat_tool import ernie_sat_web
import os

class SAT:
    def __init__(self, mfa_version='v1'):
        self.mfa_version = mfa_version

    def zh_synthesize_edit(self,
                            old_str:str,
                            new_str:str,
                            input_name:os.PathLike,
                            output_name:os.PathLike,
                            task_name:str="synthesize"
                            ):

        if task_name not in ['synthesize', 'edit']:
            print("task name only in ['edit', 'synthesize']")
            return None
        
        # erniesat model
        erniesat_config = "source/model/erniesat_aishell3_ckpt_1.2.0/default.yaml"
        erniesat_ckpt = "source/model/erniesat_aishell3_ckpt_1.2.0/snapshot_iter_289500.pdz"
        erniesat_stat = "source/model/erniesat_aishell3_ckpt_1.2.0/speech_stats.npy"
        phones_dict = "source/model/erniesat_aishell3_ckpt_1.2.0/phone_id_map.txt"
        duration_adjust = True
        # vocoder
        voc = "hifigan_aishell3"
        voc_config = "source/model/hifigan_aishell3_ckpt_0.2.0/default.yaml"
        voc_ckpt = "source/model/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz"
        voc_stat = "source/model/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy"

        source_lang = "zh"
        target_lang = "zh"
        wav_path = input_name
        output_name = output_name
        
        output_name = ernie_sat_web(erniesat_config,
                                    old_str,
                                    new_str,
                                    source_lang,
                                    target_lang, 
                                    task_name,
                                    erniesat_ckpt,
                                    erniesat_stat,
                                    phones_dict,
                                    voc_config,
                                    voc,
                                    voc_ckpt,
                                    voc_stat,
                                    duration_adjust,
                                    wav_path,
                                    output_name,
                                    mfa_version=self.mfa_version               
                                )
        return output_name


    def crossclone(self,
                    old_str:str,
                    new_str:str,input_name:os.PathLike,
                    output_name:os.PathLike,
                    source_lang:str,
                    target_lang:str,
                    ):
        # erniesat model
        erniesat_config = "source/model/erniesat_aishell3_vctk_ckpt_1.2.0/default.yaml"
        erniesat_ckpt = "source/model/erniesat_aishell3_vctk_ckpt_1.2.0/snapshot_iter_489000.pdz"
        erniesat_stat = "source/model/erniesat_aishell3_vctk_ckpt_1.2.0/speech_stats.npy"
        phones_dict = "source/model/erniesat_aishell3_vctk_ckpt_1.2.0/phone_id_map.txt"
        duration_adjust = True
        # vocoder
        voc = "hifigan_aishell3"
        voc_config = "source/model/hifigan_aishell3_ckpt_0.2.0/default.yaml"
        voc_ckpt = "source/model/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz"
        voc_stat = "source/model/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy"

        task_name = 'synthesize'
        wav_path = input_name
        output_name = output_name
        
        output_name = ernie_sat_web(erniesat_config,
                                    old_str,
                                    new_str,
                                    source_lang,
                                    target_lang, 
                                    task_name,
                                    erniesat_ckpt,
                                    erniesat_stat,
                                    phones_dict,
                                    voc_config,
                                    voc,
                                    voc_ckpt,
                                    voc_stat,
                                    duration_adjust,
                                    wav_path,
                                    output_name,
                                    mfa_version=self.mfa_version               
                                )
        return output_name

    def en_synthesize_edit(self,
                            old_str:str,
                            new_str:str,input_name:os.PathLike,
                            output_name:os.PathLike,
                            task_name:str="synthesize"):
                # erniesat model
        erniesat_config = "source/model/erniesat_vctk_ckpt_1.2.0/default.yaml"
        erniesat_ckpt = "source/model/erniesat_vctk_ckpt_1.2.0/snapshot_iter_199500.pdz"
        erniesat_stat = "source/model/erniesat_vctk_ckpt_1.2.0/speech_stats.npy"
        phones_dict = "source/model/erniesat_vctk_ckpt_1.2.0/phone_id_map.txt"
        duration_adjust = True
        # vocoder
        voc = "hifigan_aishell3"
        voc_config = "source/model/hifigan_vctk_ckpt_0.2.0/default.yaml"
        voc_ckpt = "source/model/hifigan_vctk_ckpt_0.2.0/snapshot_iter_2500000.pdz"
        voc_stat = "source/model/hifigan_vctk_ckpt_0.2.0/feats_stats.npy"

        source_lang = "en"
        target_lang = "en"
        wav_path = input_name
        output_name = output_name
        
        output_name = ernie_sat_web(erniesat_config,
                                    old_str,
                                    new_str,
                                    source_lang,
                                    target_lang, 
                                    task_name,
                                    erniesat_ckpt,
                                    erniesat_stat,
                                    phones_dict,
                                    voc_config,
                                    voc,
                                    voc_ckpt,
                                    voc_stat,
                                    duration_adjust,
                                    wav_path,
                                    output_name,
                                    mfa_version=self.mfa_version               
                                )
        return output_name




if __name__ == '__main__':

    sat = SAT(mfa_version='v2')
    # 中文语音克隆
    print("######## 中文语音克隆 #######")
    old_str = "请播放歌曲小苹果。"
    new_str = "歌曲真好听。"
    input_name = "source/wav/SAT/upload/SSB03540307.wav"
    output_name = "source/wav/SAT/out/sat_syn.wav"
    output_name = os.path.realpath(output_name)
    sat.zh_synthesize_edit(
        old_str=old_str,
        new_str=new_str,
        input_name=input_name,
        output_name=output_name,
        task_name="synthesize"
    )

    # 中文语音编辑
    print("######## 中文语音编辑 #######")
    old_str = "今天天气很好"
    new_str = "今天心情很好"
    input_name = "source/wav/SAT/upload/SSB03540428.wav"
    output_name = "source/wav/SAT/out/sat_edit.wav"
    output_name = os.path.realpath(output_name)
    print(os.path.realpath(output_name))
    sat.zh_synthesize_edit(
        old_str=old_str,
        new_str=new_str,
        input_name=input_name,
        output_name=output_name,
        task_name="edit"
    )

    # 中文跨语言克隆
    print("######## 中文 跨语言音色克隆 #######")
    old_str = "请播放歌曲小苹果。"
    new_str = "Thank you very mych! what can i do for you"
    source_lang='zh'
    target_lang='en'
    input_name = "source/wav/SAT/upload/SSB03540307.wav"
    output_name = "source/wav/SAT/out/sat_cross_zh2en.wav"
    output_name = os.path.realpath(output_name)
    print(os.path.realpath(output_name))
    sat.crossclone(
        old_str=old_str,
        new_str=new_str,
        input_name=input_name,
        output_name=output_name,
        source_lang=source_lang,
        target_lang=target_lang
    )

    # 英文跨语言克隆
    print("######## 英文 跨语言音色克隆 #######")
    old_str = "For that reason cover should not be given."
    new_str = "今天天气很好"
    source_lang='en'
    target_lang='zh'
    input_name = "source/wav/SAT/upload/p243_313.wav"
    output_name = "source/wav/SAT/out/sat_cross_en2zh.wav"
    output_name = os.path.realpath(output_name)
    print(os.path.realpath(output_name))
    sat.crossclone(
        old_str=old_str,
        new_str=new_str,
        input_name=input_name,
        output_name=output_name,
        source_lang=source_lang,
        target_lang=target_lang
    )

    # 英文语音克隆
    print("######## 英文音色克隆 #######")
    old_str = "For that reason cover should not be given."
    new_str = "I love you very much do you love me"
    input_name = "source/wav/SAT/upload/p243_313.wav"
    output_name = "source/wav/SAT/out/sat_syn_en.wav"
    output_name = os.path.realpath(output_name)
    sat.en_synthesize_edit(
        old_str=old_str,
        new_str=new_str,
        input_name=input_name,
        output_name=output_name,
        task_name="synthesize"
    )

    # 英文语音编辑
    print("######## 英文语音编辑 #######")
    old_str = "For that reason cover should not be given."
    new_str = "For that reason cover is not impossible to be given."
    input_name = "source/wav/SAT/upload/p243_313.wav"
    output_name = "source/wav/SAT/out/sat_edit_en.wav"
    output_name = os.path.realpath(output_name)
    sat.en_synthesize_edit(
        old_str=old_str,
        new_str=new_str,
        input_name=input_name,
        output_name=output_name,
        task_name="edit"
    )

