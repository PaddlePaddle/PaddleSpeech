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
from config import TOP_K
from encode import get_audio_embedding
from logs import LOGGER


def do_search(host, table_name, audio_path, milvus_cli, mysql_cli):
    """
    Search the uploaded audio in Milvus/MySQL
    """
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        feat = get_audio_embedding(audio_path)
        vectors = milvus_cli.search_vectors(table_name, [feat], TOP_K)
        vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        for i in range(len(paths)):
            tmp = "http://" + str(host) + "/data?audio_path=" + str(paths[i])
            paths[i] = tmp
            distances[i] = (1 - distances[i]) * 100
        return vids, paths, distances
    except Exception as e:
        LOGGER.error(f"Error with search: {e}")
        sys.exit(1)
