from paddlespeech.t2s.models.CosyVoice.cosyvoice import CosyVoice2
import sys
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import paddle
import torch
paddle.seed(42)
from paddlespeech.t2s.frontend.CosyVoiceFrontEnd.frontend import CosyVoiceFrontEnd
from paddlespeech.t2s.models.CosyVoice.llm import Qwen2LM,Qwen2Encoder
from paddlespeech.t2s.models.CosyVoice.common import ras_sampling
from paddlespeech.t2s.frontend.CosyVoiceFrontEnd.tokenizer import get_qwen_tokenizer
from hyperpyyaml import load_hyperpyyaml
hyper_yaml_path = "/root/paddlejob/workspace/zhangjinghong/CosyVoice/pretrained_models/CosyVoice2-0.5B_paddle/cosyvoice2.yaml"
with open(hyper_yaml_path, 'r') as f:
    configs = load_hyperpyyaml(f)

# frontend = CosyVoiceFrontEnd(
#     lambda:get_qwen_tokenizer('/root/paddlejob/workspace/zhangjinghong/CosyVoice/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN',skip_special_tokens=True),
#     configs['feat_extractor'],
#     "/root/paddlejob/workspace/zhangjinghong/CosyVoice/pretrained_models/CosyVoice2-0.5B/campplus.onnx",
#     "/root/paddlejob/workspace/zhangjinghong/CosyVoice/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx",
#     "/root/paddlejob/workspace/zhangjinghong/CosyVoice/pretrained_models/CosyVoice2-0.5B/spk2info.pt",
#     configs['allowed_special']
#     )
# prompt_wav = "../CosyVoice/gaoziyuan_10.wav"
# tts_text = frontend.text_normalize("定位解决了云存储服务和 data loader 等多个环节的性能波动问题", split=True, text_frontend=True)
# prompt_text = frontend.text_normalize('清晨的阳光透过树叶洒在地面上，微风轻轻吹过，带来花草的香气。街边的咖啡店刚开门，传来阵阵烘焙的香味，让人感到放松与愉快。', split=True, text_frontend=True)
# model_input = frontend.frontend_zero_shot(tts_text, prompt_text, prompt_wav, 24000,'')
# paddle.save(model_input,'model_input.pdparams')
# # cosyvoice_model = CosyVoice2("../CosyVoice/pretrained_models/CosyVoice2-0.5B_paddle")
# model = AutoModelForCausalLM.from_pretrained('/root/paddlejob/workspace/zhangjinghong/test/pretrained/Qwen/Qwen2-0.5B')
# print(type(model))
# llm = Qwen2Encoder(model)
# qwen_lm = Qwen2LM(896,896,6561,llm,ras_sampling)
# state_dict = paddle.load("/root/paddlejob/workspace/zhangjinghong/CosyVoice/pretrained_models/CosyVoice2-0.5B_paddle/llm.pdparams")
# qwen_lm.set_state_dict(state_dict)


# new_dict = torch.load("/root/paddlejob/workspace/zhangjinghong/CosyVoice/data.pt")
# text = new_dict['text'] 
# text_len = new_dict['text_len']
# prompt_text = new_dict['prompt_text']
# prompt_text_len = new_dict['prompt_text_len']
# prompt_speech_token = new_dict['prompt_speech_token']
# prompt_speech_token_len = new_dict['prompt_speech_token_len']
# embedding = new_dict['embedding']
# uuid = new_dict['uuid']


# text = model_input['text'] 
# text_len = model_input['text_len']
# prompt_text = model_input['prompt_text']
# prompt_text_len = model_input['prompt_text_len']
# prompt_speech_token = model_input['llm_prompt_speech_token']
# prompt_speech_token_len = model_input['llm_prompt_speech_token_len']
# embedding = model_input['llm_embedding']
# uuid = new_dict['uuid']

# # 统一设备并转换为Paddle张量
# device = paddle.CUDAPlace(0)  # 使用GPU设备
# text_tensor = paddle.to_tensor(text).cuda()
# prompt_text_tensor = paddle.to_tensor(prompt_text).cuda()
# prompt_speech_token_tensor = paddle.to_tensor(prompt_speech_token).cuda()
# embedding_tensor = paddle.to_tensor(embedding, dtype='float32').cuda()
# # 确保长度张量也统一设备并正确转换
# text_len_tensor = text_len.cuda() if hasattr(text_len, 'cuda') else paddle.to_tensor(text_len).cuda()
# prompt_text_len_tensor = prompt_text_len.cuda() if hasattr(prompt_text_len, 'cuda') else paddle.to_tensor(prompt_text_len).cuda()
# prompt_speech_token_len_tensor = prompt_speech_token_len.cuda() if hasattr(prompt_speech_token_len, 'cuda') else paddle.to_tensor(prompt_speech_token_len).cuda()
# token=[]
# for i in qwen_lm.inference(text=text_tensor,
#     text_len=text_len_tensor,
#     prompt_text=prompt_text_tensor,
#     prompt_text_len=prompt_text_len_tensor,
#     prompt_speech_token=prompt_speech_token_tensor,
#     prompt_speech_token_len=prompt_speech_token_len_tensor,
#     embedding=embedding_tensor,
#     uuid=uuid):
#     token.append(i)
#     # print(text)
#     print("token: ",i)

############################################################################################################################


flow = configs['flow']
flow_state_dict = paddle.load("/root/paddlejob/workspace/zhangjinghong/CosyVoice/pretrained_models/CosyVoice2-0.5B_paddle/flow.pdparams")
flow.set_state_dict(flow_state_dict)
input_dict = torch.load("/root/paddlejob/workspace/zhangjinghong/test/CosyVoice/data.pt")
flow.eval()
tts_mel, _ = flow.inference(
    token = paddle.to_tensor(input_dict['token']),
    token_len = paddle.to_tensor(input_dict['token_len']),
    prompt_token = paddle.to_tensor(input_dict['prompt_token'].cpu().numpy(), dtype = 'int32'),
    prompt_token_len = paddle.to_tensor(input_dict['prompt_token_len'].cpu().numpy()),
    prompt_feat = paddle.to_tensor(input_dict['prompt_feat'].cpu().numpy()),
    prompt_feat_len = paddle.to_tensor(input_dict['prompt_feat_len'].cpu().numpy()),
    embedding = paddle.to_tensor(input_dict['embedding'].cpu().numpy()),
    streaming = input_dict['streaming'],
    finalize = input_dict['finalize']
)
paddle.save(tts_mel,"tts_mel.pdparams")

############################################################################################################################

from paddlespeech.t2s.models.hifigan.cosy_hifigan import HiFTGenerator
from paddlespeech.t2s.models.hifigan.f0_predictor import ConvRNNF0Predictor
hift_state_dict = paddle.load("/root/paddlejob/workspace/zhangjinghong/CosyVoice/pretrained_models/CosyVoice2-0.5B_paddle/hift.pdparams")
input_mel = paddle.to_tensor(torch.load("../CosyVoice/tts_mel.pt").detach().cpu().numpy()).cuda()
hift_cache_source= paddle.to_tensor(torch.load("../CosyVoice/hift_cache_source.pt").detach().cpu().numpy()).cuda()
# hift_cache_source = paddle.zeros([1, 1, 0])
hift_configs = configs['hift']
f0_config = configs['f0_predictor']
f0_predictor = ConvRNNF0Predictor(**f0_config)

hift_configs['f0_predictor'] = f0_predictor
hift = HiFTGenerator(**hift_configs)
hift.set_state_dict(hift_state_dict)
# for k,v in hift.state_dict().items():
#     print(k,v.shape)
# print("---"*40)
for k,v in hift_state_dict.items():
    print(k,v.shape)
tts_speech, tts_source = hift.inference(speech_feat=input_mel, cache_source=hift_cache_source)
paddle.save(tts_speech,"speech.pdparams")
# tts_speech,_ = hift.inference(input_dict['tts_mel'],input_dict['cache_source'])

import torchaudio
import torch
torchaudio.save("paddle.wav",torch.tensor(tts_speech.numpy()),24000)
# sf.write("paddle.wav",tts_speech[0],24000)