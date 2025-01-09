# MIT License, Copyright (c) 2023-Present, Descript.
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
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/ml/test_model.py)
import sys
import tempfile

import paddle
from paddle import nn
sys.path.append("../..")
from audiotools import ml
from audiotools import util
from paddlespeech.vector.training.seeding import seed_everything
SEED = 0


def seed_and_run(model, *args, **kwargs):
    seed_everything(SEED)
    return model(*args, **kwargs)


class Model(ml.BaseModel):
    def __init__(self, arg1: float=1.0):
        super().__init__()
        self.arg1 = arg1
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class OtherModel(ml.BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def test_base_model():
    # Save and load
    # ml.BaseModel.EXTERN += ["test_model"]

    x = paddle.randn([10, 1])
    model1 = Model()

    # assert str(model1.device) == 'Place(cpu)'

    out1 = seed_and_run(model1, x)

    with tempfile.NamedTemporaryFile(suffix=".pdparams") as f:
        model1.save(
            f.name, )
        model2 = Model.load(f.name)
        out2 = seed_and_run(model2, x)
        assert paddle.allclose(out1, out2)

        # test re-export
        model2.save(f.name)
        model3 = Model.load(f.name)
        out3 = seed_and_run(model3, x)
        assert paddle.allclose(out1, out3)

        # make sure legacy/save load works
        model1.save(f.name, package=False)
        model2 = Model.load(f.name)
        out2 = seed_and_run(model2, x)
        assert paddle.allclose(out1, out2)

        # make sure new way -> legacy save -> legacy load works
        model1.save(f.name, package=False)
        model2 = Model.load(f.name)
        model2.save(f.name, package=False)
        model3 = Model.load(f.name)
        out3 = seed_and_run(model3, x)

        # save/load without package, but with model2 being a model
        # without an argument of arg1 to its instantiation.
        model1.save(f.name, package=False)
        model2 = OtherModel.load(f.name)
        out2 = seed_and_run(model2, x)
        assert paddle.allclose(out1, out2)

        assert paddle.allclose(out1, out3)

    with tempfile.TemporaryDirectory() as d:
        model1.save_to_folder(d, {"data": 1.0})
        Model.load_from_folder(d)
