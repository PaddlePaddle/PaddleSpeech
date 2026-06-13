import paddle

from ..utils import deprecate
from ..utils.import_utils import is_torch_npu_available

if is_torch_npu_available():
    import torch_npu
ACTIVATION_FUNCTIONS = {
    "swish": paddle.nn.SiLU(),
    "silu": paddle.nn.SiLU(),
    "mish": paddle.nn.Mish(),
    "gelu": paddle.nn.GELU(),
    "relu": paddle.nn.ReLU(),
}


def get_activation(act_fn: str) -> paddle.nn.Layer:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class FP32SiLU(paddle.nn.Layer):
    """
    SiLU activation function with input upcasted to torch.float32.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.silu(inputs.float(), inplace=False).to(inputs.dtype)


class GELU(paddle.nn.Layer):
    """
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True
    ):
        super().__init__()
        self.proj = paddle.nn.Linear(
            in_features=dim_in, out_features=dim_out, bias_attr=bias
        )
        self.approximate = approximate

    def gelu(self, gate: paddle.Tensor) -> paddle.Tensor:
        if gate.device.type != "mps":
            return paddle.nn.functional.gelu(gate, approximate=self.approximate)
        return paddle.nn.functional.gelu(
            gate.to(dtype=paddle.float32), approximate=self.approximate
        ).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(paddle.nn.Layer):
    """
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = paddle.nn.Linear(
            in_features=dim_in, out_features=dim_out * 2, bias_attr=bias
        )

    def gelu(self, gate: paddle.Tensor) -> paddle.Tensor:
        if gate.device.type != "mps":
            return paddle.nn.functional.gelu(gate)
        return paddle.nn.functional.gelu(gate.to(dtype=paddle.float32)).to(
            dtype=gate.dtype
        )

    def forward(self, hidden_states, *args, **kwargs):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        hidden_states = self.proj(hidden_states)
        if is_torch_npu_available():
            return torch_npu.npu_geglu(hidden_states, dim=-1, approximate=1)[0]
        else:
            hidden_states, gate = hidden_states.chunk(2, dim=-1)
            return hidden_states * self.gelu(gate)


class ApproximateGELU(paddle.nn.Layer):
    """
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = paddle.nn.Linear(
            in_features=dim_in, out_features=dim_out, bias_attr=bias
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.proj(x)
        return x * paddle.nn.functional.sigmoid(1.702 * x)
