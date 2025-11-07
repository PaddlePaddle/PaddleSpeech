from paddlespeech.t2s.models.CosyVoice.cosyvoice import CosyVoice2
import sys
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import paddle
import torch
from paddlespeech.t2s.models.CosyVoice.llm import Qwen2LM,ras_sampling,Qwen2Encoder
# cosyvoice_model = CosyVoice2("../CosyVoice/pretrained_models/CosyVoice2-0.5B_paddle")
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B')
llm = Qwen2Encoder(model)
qwen_lm = Qwen2LM(896,896,6561,llm,ras_sampling)
state_dict = paddle.load("/root/paddlejob/workspace/zhangjinghong/CosyVoice/pretrained_models/CosyVoice2-0.5B_paddle/llm.pdparams")
qwen_lm.set_state_dict(state_dict)
new_dict = torch.load("data.pt")
text = new_dict['text'] 
text_len = new_dict['text_len']
prompt_text = new_dict['prompt_text']
prompt_text_len = new_dict['prompt_text_len']
prompt_speech_token = new_dict['prompt_speech_token']
prompt_speech_token_len = new_dict['prompt_speech_token_len']
embedding = new_dict['embedding']
uuid = new_dict['uuid']
print("text:",text)
# for i in qwen_lm.inference(text=paddle.to_tensor(text),
#     text_len=text_len,
#     prompt_text=paddle.to_tensor(prompt_text),
#     prompt_text_len=prompt_text_len,
#     prompt_speech_token=paddle.to_tensor(prompt_speech_token),
#     prompt_speech_token_len=prompt_speech_token_len,
#     embedding=paddle.to_tensor(embedding,dtype = 'float32'),
#     uuid=uuid):
#     print(text)
#     print(i)

