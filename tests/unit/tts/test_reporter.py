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
import numpy as np

from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.reporter import scope
from paddlespeech.t2s.training.reporter import Summary


def test_reporter_scope():
    first = {}
    second = {}
    third = {}

    with scope(first):
        report("first_begin", 1)
        with scope(second):
            report("second_begin", 2)
            with scope(third):
                report("third_begin", 3)
                report("third_end", 4)
            report("seconf_end", 5)
        report("first_end", 6)

    assert first == {'first_begin': 1, 'first_end': 6}
    assert second == {'second_begin': 2, 'seconf_end': 5}
    assert third == {'third_begin': 3, 'third_end': 4}
    print(first)
    print(second)
    print(third)


def test_summary():
    summary = Summary()
    summary.add(1)
    summary.add(2)
    summary.add(3)
    state = summary.make_statistics()
    print(state)
    np.testing.assert_allclose(np.array(list(state)),
                               np.array([2.0, np.std([1, 2, 3])]))
