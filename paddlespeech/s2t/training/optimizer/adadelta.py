# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import framework
from paddle.optimizer import Optimizer

__all__ = []


class SimpleAdadelta(Optimizer):
    r"""
    **Notes: This API does not support sparse parameter optimization.**

    Adadelta Optimizer. Please refer to this for details:
    `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_.

    The update is done as follows:

    .. math::

        E(g_t^2) &= \rho * E(g_{t-1}^2) + (1-\rho) * g^2

        learning\_rate &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }

        E(dx_t^2) &= \rho * E(dx_{t-1}^2) + (1-\rho) * (-g*learning\_rate)^2

    Args:
        learning_rate (float|Tensor|LearningRateDecay, optional): The learning rate used to update ``Parameter``.
            It can be a float value, a ``Tensor`` with a float type or a LearningRateDecay. The default value is 0.001.
        epsilon (float): a small float number for numeric stability. Default 1.0e-6.
        rho (float): a floating point value indicating the decay rate. Default 0.95.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in paramter groups \
            represents the scale of base learning_rate. \
            The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
            It canbe a float value as coeff of L2 regularization or \
            :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        foreach (bool, optional): whether foreach implementation of optimizer is used. The default value is None.
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            The default value is False.
        name (str, optional): The default value is None. Normally there is no need for user
                to set this property. For more information, please refer to
                :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            import paddle
            from paddlespeech.s2t.training.optimizer.adadelta import SimpleAdadelta

            inp = paddle.uniform([10, 10], dtype="float32", min=-0.1, max=0.1)
            linear = paddle.nn.Linear(10, 10)
            out = linear(inp)
            loss = paddle.mean(out)
            adadelta = SimpleAdadelta(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
            out.backward()
            adadelta.step()
            adadelta.clear_grad()

    """

    def __init__(
            self,
            learning_rate=0.001,
            epsilon=1.0e-6,
            rho=0.95,
            parameters=None,
            weight_decay=0.0,
            foreach=None,
            maximize=False,
            name=None, ):
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")
        if epsilon is None:
            raise ValueError("epsilon is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        super(SimpleAdadelta, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            name=name, )

        self._epsilon = epsilon
        self._rho = rho

        self.state = 0  # self.state is 0 or 1, use to control init square_avgs and acc_deltas
        self._weight_decay = weight_decay
        self._learning_rate = learning_rate
        self._foreach = foreach
        self._maximize = maximize
        self.square_avgs = []
        self.acc_deltas = []

    @paddle.no_grad()
    @framework.dygraph_only
    def step(self):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not isinstance(self._parameter_list[0], dict):
            params_grads = []
            for param in self._parameter_list:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()

                    params_grads.append((param, grad_var))
                    if self.state == 0:
                        self.square_avg = paddle.zeros_like(param)
                        self.acc_delta = paddle.zeros_like(param)
                        self.square_avgs.append(self.square_avg)
                        self.acc_deltas.append(self.acc_delta)

        else:
            # optimize parameters in groups
            params_grads = []
            for idx, param_group in enumerate(self._param_groups):
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        params_grads.append((param, grad_var))
                        if self.state == 0:
                            self.square_avg = paddle.zeros_like(param)
                            self.acc_delta = paddle.zeros_like(param)
                            self.square_avgs.append(self.square_avg)
                            self.acc_deltas.append(self.acc_delta)

        self.state = 1
        adadelta(
            params_grads,
            square_avgs=self.square_avgs,
            acc_deltas=self.acc_deltas,
            learning_rate=self._learning_rate,
            rho=self._rho,
            epsilon=self._epsilon,
            weight_decay=self._weight_decay,
            foreach=self._foreach,
            maximize=self._maximize)


def adadelta(params_grads,
             square_avgs,
             acc_deltas,
             foreach=None,
             *,
             learning_rate: float,
             rho: float,
             epsilon: float,
             weight_decay: float,
             maximize: bool):

    if foreach is None:
        # if foreach is None, set False
        foreach = False
    if not foreach:
        # optimizer is used
        func = _single_tensor_adadelta

    func(
        params_grads,
        square_avgs,
        acc_deltas,
        learning_rate=learning_rate,
        rho=rho,
        epsilon=epsilon,
        weight_decay=weight_decay,
        maximize=maximize)


def _single_tensor_adadelta(params_grads,
                            square_avgs,
                            acc_deltas,
                            *,
                            learning_rate: float,
                            rho: float,
                            epsilon: float,
                            weight_decay: float,
                            maximize: bool):
    """
    Calculate variables(square_avgs, acc_deltas) and update parameters.
    """

    for (params_grad, square_avg, acc_delta) in zip(params_grads, square_avgs,
                                                    acc_deltas):
        param, grad = params_grad
        grad = grad if not maximize else -grad
        if weight_decay != 0:
            grad.set_value(grad.add(paddle.multiply(param, weight_decay)))

        if paddle.is_complex(param):
            square_avg = paddle.as_real(square_avg)
            acc_delta = paddle.as_real(acc_delta)
            grad = paddle.as_real(grad)
        # square_avg = square_avg * rho + (1-rho) * grad * grad
        square_avg.set_value(
            paddle.multiply(square_avg, paddle.to_tensor(rho)).add(
                paddle.multiply(paddle.to_tensor(1 - rho), grad.square())))
        # std = (square_avg + eps).sqrt()
        std = square_avg.add(paddle.to_tensor(epsilon)).sqrt_()
        # delta = std / (acc_delta + eps).sqrt() * grad
        delta = (paddle.multiply(
            paddle.divide(
                acc_delta.add(paddle.to_tensor(epsilon)).sqrt_(), std), grad))
        # acc_delta = acc_delta * rho + (1-rho) * delta * delta
        acc_delta.set_value(
            paddle.multiply(acc_delta, paddle.to_tensor(rho)).add(
                paddle.multiply(paddle.to_tensor(1 - rho), delta.square())))
        if paddle.is_complex(param):
            delta = paddle.as_real(delta)
        # param = param - delta*learning_rate
        param.set_value(
            param.add(
                paddle.multiply(
                    delta.astype('float32'), paddle.to_tensor(-learning_rate))))
