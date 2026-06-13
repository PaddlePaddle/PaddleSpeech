# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import paddle

"""Swish() activation function for Conformer."""


class Swish(paddle.nn.Layer):
    """Construct an Swish object."""

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Return Swish activation function."""
        return x * paddle.nn.functional.sigmoid(x)


class Snake(paddle.nn.Layer):
    '''
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = Snake(256)
        >>> x = paddle.randn([1, 256, 100])  # Example input
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: number of input features (channel dimension)
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        '''
        super(Snake, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        
        # 避免除零的小常数
        self.no_div_by_zero = 1e-9

        # 初始化alpha的值：对数尺度下初始为0，线性尺度下初始为alpha
        if self.alpha_logscale:
            initial_value = 0.0  # 对数尺度下，初始化为0，前向传播中会进行exp运算
        else:
            initial_value = alpha  # 线性尺度下，直接初始化为alpha

        # 创建可训练参数alpha - 使用PaddlePaddle的方式
        # 注意：这里使用self.create_parameter而不是paddle.create_parameter
        self.alpha = self.create_parameter(
            shape=[in_features],  # 参数形状为[in_features]
            dtype='float32',       # 数据类型
            default_initializer=paddle.nn.initializer.Constant(value=initial_value)  # 初始化器
        )
        
        # 设置参数是否需要梯度更新（是否可训练）
        # 在PaddlePaddle中，通过设置stop_gradient来控制
        self.alpha.stop_gradient = not alpha_trainable

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        '''
        # 调整alpha的维度以匹配输入x: [B, C, T] -> alpha需要变为[1, C, 1]
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # 从[C]变为[1, C, 1]
        
        # 如果使用对数尺度，对alpha取指数
        if self.alpha_logscale:
            alpha = paddle.exp(alpha)
        
        # 计算Snake激活函数
        # 公式: x + (1.0 / (alpha + epsilon)) * sin(x * alpha)^2
        sin_term = paddle.sin(x * alpha)
        result = x + (1.0 / (alpha + self.no_div_by_zero)) * (sin_term ** 2)
        
        return result
