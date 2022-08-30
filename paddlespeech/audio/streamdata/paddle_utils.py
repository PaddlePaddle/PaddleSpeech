#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
# Modified from https://github.com/webdataset/webdataset
#
"""Mock implementations of paddle interfaces when paddle is not available."""

try:
    from paddle.io import DataLoader, IterableDataset
except ModuleNotFoundError:

    class IterableDataset:
        """Empty implementation of IterableDataset when paddle is not available."""

        pass

    class DataLoader:
        """Empty implementation of DataLoader when paddle is not available."""

        pass


try:
    from paddle import Tensor as PaddleTensor
except ModuleNotFoundError:

    class TorchTensor:
        """Empty implementation of PaddleTensor when paddle is not available."""

        pass
