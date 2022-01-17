#!/usr/bin/python3
#! coding:utf-8

import paddle

def test_filterbank(device):
    from paddleaudio.features.core import compute_fbank_matrix
    paddle.device.set_device(device)
    inputs = paddle.zeros([10, 101, 201])
    print(inputs)
    # fbanks = compute_fbank_matrix(inputs)

    # fbanks = torch.zeros([1, 1, 1], device=device)
    
    # input1 = paddle.rand([1, 101, 201], device=device) * 10
    # input2 = paddle.rand([1, 101, 201], device=device) 
    # input3 = paddle.cat([input1, input2], dim=0)
    # fbank1 = compute_fbank_matrix(input1)
    # fbank2 = compute_fbank_matrix(input2)
    # fbank3 = compute_fbank_matrix(input3)



test_filterbank("cpu")
