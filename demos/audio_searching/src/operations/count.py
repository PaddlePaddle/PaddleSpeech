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
import sys

from config import DEFAULT_TABLE
from logs import LOGGER


def do_count(table_name, milvus_cli):
    """
    Returns the total number of vectors in the system
    """
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return None
        num = milvus_cli.count(table_name)
        return num
    except Exception as e:
        LOGGER.error(f"Error attempting to count table {e}")
        sys.exit(1)


def do_count_vpr(table_name, mysql_cli):
    """
    Returns the total number of spk in the system
    """
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        num = mysql_cli.count_table(table_name)
        return num
    except Exception as e:
        LOGGER.error(f"Error attempting to count table {e}")
        sys.exit(1)


def do_list(table_name, mysql_cli):
    """
    Returns the total records of vpr in the system
    """
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        spk_ids, audio_paths, _ = mysql_cli.list_vpr(table_name)
        return spk_ids, audio_paths
    except Exception as e:
        LOGGER.error(f"Error attempting to count table {e}")
        sys.exit(1)


def do_get(table_name, spk_id, mysql_cli):
    """
    Returns the audio path by spk_id in the system
    """
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        audio_apth = mysql_cli.search_audio_vpr(table_name, spk_id)
        return audio_apth
    except Exception as e:
        LOGGER.error(f"Error attempting to count table {e}")
        sys.exit(1)
