import paddle
import torch

torch_model_dict = torch.load('large-v3-turbo.pt')['model_state_dict']

paddle_model_state_dict = {}
for key, val in torch_model_dict.items():
    if key.endswith(
            'weight'
    ) and val.ndim == 2 and key != "decoder.token_embedding.weight":
        val = val.T
    paddle_model_state_dict[key] = paddle.to_tensor(
        val.cpu().numpy()).astype("float32")

# add other params in case if need, such as:
paddle_model_state_dict['dims'] = torch.load('large-v3-turbo.pt')['dims']

paddle.save(paddle_model_state_dict, 'weights.params')
