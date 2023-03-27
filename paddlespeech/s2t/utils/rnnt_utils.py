import paddle


def add_blank(ys_pad: paddle.Tensor, blank: int,
              ignore_id: int) -> paddle.Tensor:
    """ Prepad blank for transducer predictor

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        blank (int): index of <blank>

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> blank = 0
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,   4,   5],
                [ 4,  5,  6,  -1,  -1],
                [ 7,  8,  9,  -1,  -1]], dtype=torch.int32)
        >>> ys_in = add_blank(ys_pad, 0, -1)
        >>> ys_in
        tensor([[0,  1,  2,  3,  4,  5],
                [0,  4,  5,  6,  0,  0],
                [0,  7,  8,  9,  0,  0]])
    """
    bs = ys_pad.shape[0]
    _blank = paddle.to_tensor([blank], dtype=paddle.long, stop_gradient=True)
    _blank = _blank.repeat(bs).unsqueeze(1)  # [bs,1]
    out = paddle.concat([_blank, ys_pad], axis=1)  # [bs, Lmax+1]
    return paddle.where(out == ignore_id, blank, out)


def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import

    activation_funcs = {
        "hardtanh": paddle.nn.Hardtanh,
        "tanh": paddle.nn.Tanh,
        "relu": paddle.nn.ReLU,
        "selu": paddle.nn.SELU,
        "silu": paddle.nn.Silu,
        "gelu": paddle.nn.GELU
    }

    return activation_funcs[act]()


def get_rnn(rnn_type: str):
    assert rnn_type in ["rnn", "lstm", "gru"]
    if rnn_type == "rnn":
        return paddle.nn.RNN
    elif rnn_type == "lstm":
        return paddle.nn.LSTM
    else:
        return paddle.nn.GRU
