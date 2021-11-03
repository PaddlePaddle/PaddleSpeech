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
import logging
import sys
from pathlib import Path
from typing import Any
from typing import Union

import h5py
import numpy as np


def read_hdf5(filename: Union[Path, str], dataset_name: str) -> Any:
    """Read a dataset from a HDF5 file.

    Parameters
    ----------
    filename : Union[Path, str]
        Path of the HDF5 file.
    dataset_name : str
        Name of the dataset to read.

    Returns
    -------
    Any
        The retrieved dataset.
    """
    filename = Path(filename)

    if not filename.exists():
        logging.error(f"There is no such a hdf5 file ({filename}).")
        sys.exit(1)

    hdf5_file = h5py.File(filename, "r")

    if dataset_name not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({dataset_name})")
        sys.exit(1)

    # [()]: a special syntax of h5py to get the dataset as-is
    hdf5_data = hdf5_file[dataset_name][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(filename: Union[Path, str],
               dataset_name: str,
               write_data: np.ndarray,
               is_overwrite: bool=True) -> None:
    """Write dataset to HDF5 file.

    Parameters
    ----------
    filename : Union[Path, str]
        Path of the HDF5 file.
    dataset_name : str
        Name of the dataset to write to.
    write_data : np.ndarrays
        The data to write.
    is_overwrite : bool, optional
        Whether to overwrite, by default True
    """
    # convert to numpy array
    filename = Path(filename)
    write_data = np.array(write_data)

    # check folder existence
    filename.parent.mkdir(parents=True, exist_ok=True)

    # check hdf5 existence
    if filename.exists():
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(filename, "r+")
        # check dataset existence
        if dataset_name in hdf5_file:
            if is_overwrite:
                logging.warning("Dataset in hdf5 file already exists. "
                                "recreate dataset in hdf5.")
                hdf5_file.__delitem__(dataset_name)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(filename, "w")

    # write data to hdf5
    hdf5_file.create_dataset(dataset_name, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()
