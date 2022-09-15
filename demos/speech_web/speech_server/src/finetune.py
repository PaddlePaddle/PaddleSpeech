# 
# GE2E 里面的函数会干扰这边的训练过程，引起错误
# 单独运行此处的 finetune 微调过程
#
import argparse
import os
import subprocess
# from src.ft.finetune_tool import finetune_model
# from ft.finetune_tool import finetune_model, synthesize

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
    def __init__(self, mfa_version='v1', pretrained_model_dir="source/model/fastspeech2_aishell3_ckpt_1.1.0"):
        self.mfa_version = mfa_version
        self.pretrained_model_dir = pretrained_model_dir

    def finetune(self, input_dir, exp_dir = 'temp', epoch=10, batch_size=2):
        
        mfa_dir = os.path.join(exp_dir, 'mfa_result')
        dump_dir = os.path.join(exp_dir, 'dump')
        output_dir = os.path.join(exp_dir, 'exp')
        lang = "zh"
        ngpu = 0

        cmd = f"""
        python src/ft/finetune_tool.py --input_dir {input_dir} \
                                        --pretrained_model_dir {self.pretrained_model_dir} \
                                        --mfa_dir {mfa_dir} \
                                        --dump_dir {dump_dir} \
                                        --output_dir {output_dir} \
                                        --lang {lang} \
                                        --ngpu {ngpu} \
                                        --epoch {epoch} \
                                        --batch_size {batch_size} \
                                        --mfa_version {self.mfa_version}
        """
        
        return self.run_cmd(cmd=cmd, output_name=exp_dir)
        

    def synthesize(self, text, wav_name, out_wav_dir, exp_dir = 'tmp_dir'):

        # 合成测试
        pretrained_model_dir = self.pretrained_model_dir
        print("exp_dir: ", exp_dir)
        dump_dir = os.path.join(exp_dir, 'dump')
        output_dir = os.path.join(exp_dir, 'exp')
        text_path = os.path.join(exp_dir, 'sentences.txt')
        lang = "zh"

        model_path = f"{output_dir}/checkpoints"
        ckpt = find_max_ckpt(model_path)

        # 生成对应的语句
        with open(text_path, "w", encoding='utf8') as f:
            f.write(wav_name+" "+text)
        
        lang = "zh"
        spk_id = 0
        ngpu = 0
        am = "fastspeech2_aishell3"
        am_config = f"{pretrained_model_dir}/default.yaml"
        am_ckpt = f"{output_dir}/checkpoints/snapshot_iter_{ckpt}.pdz"
        am_stat = f"{pretrained_model_dir}/speech_stats.npy"
        speaker_dict = f"{dump_dir}/speaker_id_map.txt"
        phones_dict  = f"{dump_dir}/phone_id_map.txt"
        tones_dict = None
        voc = "hifigan_aishell3"
        voc_config = "source/model/hifigan_aishell3_ckpt_0.2.0/default.yaml"
        voc_ckpt = "source/model/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz"
        voc_stat = "source/model/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy"
        
        cmd = f"""
        python src/ft/synthesize.py \
        --am={am} \
        --am_config={am_config} \
        --am_ckpt={am_ckpt} \
        --am_stat={am_stat} \
        --voc={voc} \
        --voc_config={voc_config} \
        --voc_ckpt={voc_ckpt} \
        --voc_stat={voc_stat} \
        --lang={lang} \
        --text={text_path}\
        --output_dir={out_wav_dir} \
        --phones_dict={phones_dict} \
        --speaker_dict={speaker_dict} \
        --ngpu {ngpu} \
        --spk_id={spk_id} 
        """
        out_wav_path = os.path.join(out_wav_dir, wav_name)
        return self.run_cmd(cmd, out_wav_path+'.wav')
    
    def run_cmd(self, cmd, output_name):
        p = subprocess.Popen(cmd, shell=True)
        res = p.wait()
        print(cmd)
        print("运行结果：", res)
        if res == 0:
            # 运行成功
            print(f"cmd 合成结果： {output_name}")
            if os.path.exists(output_name):
                return output_name
            else:
                # 合成的文件不存在
                return None
        else:
            # 运行失败
            return None

if __name__ == '__main__':    
    ft_model = FineTune(mfa_version='v2')
    
    exp_dir = os.path.realpath("tmp_dir/finetune")
    input_dir = os.path.realpath("source/wav/finetune/default")
    output_dir = os.path.realpath("source/wav/finetune/out")
    
    #################################
    ######## 试验轮次验证 #############
    #################################
    lab = 1
    # 先删除数据
    cmd = f"rm -rf {exp_dir}"
    os.system(cmd)
    ft_model.finetune(input_dir=input_dir, exp_dir = exp_dir, epoch=10, batch_size=2)    
    
    # 合成
    text = "今天的天气真不错"
    wav_name = "demo" + str(lab) + "_a"
    out_wav_dir = os.path.realpath("source/wav/finetune/out")
    ft_model.synthesize(text, wav_name, out_wav_dir, exp_dir = exp_dir)

