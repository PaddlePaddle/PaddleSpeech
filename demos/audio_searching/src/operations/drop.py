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


def do_drop(table_name, milvus_cli, mysql_cli):
    """
    Delete the collection of Milvus and MySQL
    """
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return "Collection is not exist"
        status = milvus_cli.delete_collection(table_name)
        mysql_cli.delete_table(table_name)
        return status
    except Exception as e:
        LOGGER.error(f"Error attempting to drop table: {e}")
        sys.exit(1)


def do_drop_vpr(table_name, mysql_cli):
    """
    Delete the table of MySQL
    """
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        mysql_cli.delete_table(table_name)
        return "OK"
    except Exception as e:
        LOGGER.error(f"Error attempting to drop table: {e}")
        sys.exit(1)


def do_delete(table_name, spk_id, mysql_cli):
    """
    Delete a record by spk_id in MySQL
    """
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        mysql_cli.delete_data_vpr(table_name, spk_id)
        return "OK"
    except Exception as e:
        LOGGER.error(f"Error attempting to drop table: {e}")
        sys.exit(1)
