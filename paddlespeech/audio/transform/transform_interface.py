# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from espnet(https://github.com/espnet/espnet)


class TransformInterface:
    """Transform Interface"""
    def __call__(self, x):
        raise NotImplementedError("__call__ method is not implemented")

    @classmethod
    def add_arguments(cls, parser):
        return parser

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Identity(TransformInterface):
    """Identity Function"""
    def __call__(self, x):
        return x
