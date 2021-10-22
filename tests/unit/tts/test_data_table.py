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
from parakeet.datasets.data_tabel import DataTable


def test_audio_dataset():
    metadata = [{'name': 'Sonic', 'v': 1000}, {'name': 'Prestol', 'v': 2000}]
    converters = {'v': lambda x: x / 1000}
    dataset = DataTable(metadata, fields=['v'], converters=converters)
    assert dataset[0] == {'v': 1.0}
