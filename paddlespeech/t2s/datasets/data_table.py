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
from multiprocessing import Manager
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

from paddle.io import Dataset


class DataTable(Dataset):
    """Dataset to load and convert data for general purpose.
    Args:
        data (List[Dict[str, Any]]): Metadata, a list of meta datum, each of which is composed of  several fields
        fields (List[str], optional): Fields to use, if not specified, all the fields in the data are used, by default None
        converters (Dict[str, Callable], optional): Converters used to process each field, by default None
        use_cache (bool, optional): Whether to use cache, by default False

    Raises:
        ValueError:
            If there is some field that does not exist in data. 
        ValueError:
            If there is some field in converters that does not exist in fields.
    """

    def __init__(self,
                 data: List[Dict[str, Any]],
                 fields: List[str]=None,
                 converters: Dict[str, Callable]=None,
                 use_cache: bool=False):
        # metadata
        self.data = data
        assert len(data) > 0, "This dataset has no examples"

        # peak an example to get existing fields.
        first_example = self.data[0]
        fields_in_data = first_example.keys()

        # check all the requested fields exist
        if fields is None:
            self.fields = fields_in_data
        else:
            for field in fields:
                if field not in fields_in_data:
                    raise ValueError(
                        f"The requested field ({field}) is not found"
                        f"in the data. Fields in the data is {fields_in_data}")
            self.fields = fields

        # check converters
        if converters is None:
            self.converters = {}
        else:
            for field in converters.keys():
                if field not in self.fields:
                    raise ValueError(
                        f"The converter has a non existing field ({field})")
            self.converters = converters

        self.use_cache = use_cache
        if use_cache:
            self._initialize_cache()

    def _initialize_cache(self):
        self.manager = Manager()
        self.caches = self.manager.list()
        self.caches += [None for _ in range(len(self))]

    def _get_metadata(self, idx: int) -> Dict[str, Any]:
        """Return a meta-datum given an index."""
        return self.data[idx]

    def _convert(self, meta_datum: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a meta datum to an example by applying the corresponding 
        converters to each fields requested.

        Args:
            meta_datum (Dict[str, Any]): Meta datum

        Returns:
            Dict[str, Any]: Converted example
        """
        example = {}
        for field in self.fields:
            converter = self.converters.get(field, None)
            meta_datum_field = meta_datum[field]
            if converter is not None:
                converted_field = converter(meta_datum_field)
            else:
                converted_field = meta_datum_field
            example[field] = converted_field
        return example

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an example given an index.
        Args:
            idx (int): Index of the example to get

        Returns:
            Dict[str, Any]: A converted example
        """
        if self.use_cache and self.caches[idx] is not None:
            return self.caches[idx]

        meta_datum = self._get_metadata(idx)
        example = self._convert(meta_datum)

        if self.use_cache:
            self.caches[idx] = example

        return example

    def __len__(self) -> int:
        """Returns the size of the dataset.

        Returns
        -------
        int
            The length of the dataset
        """
        return len(self.data)
