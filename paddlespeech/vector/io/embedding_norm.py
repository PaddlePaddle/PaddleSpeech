# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Dict

import paddle


class InputNormalization:
    spk_dict_mean: Dict[int, paddle.Tensor]
    spk_dict_std: Dict[int, paddle.Tensor]
    spk_dict_count: Dict[int, int]

    def __init__(
            self,
            mean_norm=True,
            std_norm=True,
            norm_type="global", ):
        """Do feature or embedding mean and std norm

        Args:
            mean_norm (bool, optional): mean norm flag. Defaults to True.
            std_norm (bool, optional): std norm flag. Defaults to True.
            norm_type (str, optional): norm type. Defaults to "global".
        """
        super().__init__()
        self.training = True
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.norm_type = norm_type
        self.glob_mean = paddle.to_tensor([0], dtype="float32")
        self.glob_std = paddle.to_tensor([0], dtype="float32")
        self.spk_dict_mean = {}
        self.spk_dict_std = {}
        self.spk_dict_count = {}
        self.weight = 1.0
        self.count = 0
        self.eps = 1e-10

    def __call__(self,
                 x,
                 lengths,
                 spk_ids=paddle.to_tensor([], dtype="float32")):
        """Returns the tensor with the surrounding context.
        Args:
            x (paddle.Tensor): A batch of tensors.
            lengths (paddle.Tensor): A batch of tensors containing the relative length of each
                                    sentence (e.g, [0.7, 0.9, 1.0]). It is used to avoid
                                    computing stats on zero-padded steps.
            spk_ids (paddle.Tensor, optional): tensor containing the ids of each speaker (e.g, [0 10 6]).
                                        It is used to perform per-speaker normalization when
                                        norm_type='speaker'. Defaults to paddle.to_tensor([], dtype="float32").
        Returns:
            paddle.Tensor: The normalized feature or embedding
        """
        N_batches = x.shape[0]

        current_means = []
        current_stds = []

        for snt_id in range(N_batches):

            # Avoiding padded time steps
            # actual size is the actual time data length
            actual_size = paddle.round(lengths[snt_id] *
                                       x.shape[1]).astype("int32")
            # computing actual time data statistics
            # we extract the snt_id embedding from the x
            # and the target paddle.Tensor will reduce an 0-axis
            # so we need unsqueeze operation to recover the all axis
            current_mean, current_std = self._compute_current_stats(
                x[snt_id, 0:actual_size, ...].unsqueeze(0))
            current_means.append(current_mean)
            current_stds.append(current_std)

        if self.norm_type == "global":
            current_mean = paddle.mean(paddle.stack(current_means), axis=0)
            current_std = paddle.mean(paddle.stack(current_stds), axis=0)

            if self.norm_type == "global":

                if self.training:
                    if self.count == 0:
                        self.glob_mean = current_mean
                        self.glob_std = current_std

                    else:
                        self.weight = 1 / (self.count + 1)

                        self.glob_mean = (
                            1 - self.weight
                        ) * self.glob_mean + self.weight * current_mean

                        self.glob_std = (
                            1 - self.weight
                        ) * self.glob_std + self.weight * current_std

                    self.glob_mean.detach()
                    self.glob_std.detach()

                    self.count = self.count + 1
                x = (x - self.glob_mean) / (self.glob_std)
        return x

    def _compute_current_stats(self, x):
        """Returns the tensor with the surrounding context.

        Args:
            x (paddle.Tensor): A batch of tensors.

        Returns:
             the statistics of the data
        """
        # Compute current mean
        if self.mean_norm:
            current_mean = paddle.mean(x, axis=0).detach()
        else:
            current_mean = paddle.to_tensor([0.0], dtype="float32")

        # Compute current std
        if self.std_norm:
            current_std = paddle.std(x, axis=0).detach()
        else:
            current_std = paddle.to_tensor([1.0], dtype="float32")

        # Improving numerical stability of std
        current_std = paddle.maximum(current_std,
                                     self.eps * paddle.ones_like(current_std))

        return current_mean, current_std

    def _statistics_dict(self):
        """Fills the dictionary containing the normalization statistics.
        """
        state = {}
        state["count"] = self.count
        state["glob_mean"] = self.glob_mean
        state["glob_std"] = self.glob_std
        state["spk_dict_mean"] = self.spk_dict_mean
        state["spk_dict_std"] = self.spk_dict_std
        state["spk_dict_count"] = self.spk_dict_count

        return state

    def _load_statistics_dict(self, state):
        """Loads the dictionary containing the statistics.

        Arguments
        ---------
        state : dict
            A dictionary containing the normalization statistics.
        """
        self.count = state["count"]
        if isinstance(state["glob_mean"], int):
            self.glob_mean = state["glob_mean"]
            self.glob_std = state["glob_std"]
        else:
            self.glob_mean = state["glob_mean"]  # .to(self.device_inp)
            self.glob_std = state["glob_std"]  # .to(self.device_inp)

        # Loading the spk_dict_mean in the right device
        self.spk_dict_mean = {}
        for spk in state["spk_dict_mean"]:
            self.spk_dict_mean[spk] = state["spk_dict_mean"][spk]

        # Loading the spk_dict_std in the right device
        self.spk_dict_std = {}
        for spk in state["spk_dict_std"]:
            self.spk_dict_std[spk] = state["spk_dict_std"][spk]

        self.spk_dict_count = state["spk_dict_count"]

        return state

    def to(self, device):
        """Puts the needed tensors in the right device.
        """
        self = super(InputNormalization, self).to(device)
        self.glob_mean = self.glob_mean.to(device)
        self.glob_std = self.glob_std.to(device)
        for spk in self.spk_dict_mean:
            self.spk_dict_mean[spk] = self.spk_dict_mean[spk].to(device)
            self.spk_dict_std[spk] = self.spk_dict_std[spk].to(device)
        return self

    def save(self, path):
        """Save statistic dictionary.
    
        Args:
            path (str): A path where to save the dictionary.
        """
        stats = self._statistics_dict()
        paddle.save(stats, path)

    def _load(self, path, end_of_epoch=False, device=None):
        """Load statistic dictionary.

        Arguments
        ---------
        path : str
            The path of the statistic dictionary
        device : str, None
            Passed to paddle.load(..., map_location=device)
        """
        del end_of_epoch  # Unused here.
        stats = paddle.load(path, map_location=device)
        self._load_statistics_dict(stats)
