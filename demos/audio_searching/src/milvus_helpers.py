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

from config import METRIC_TYPE
from config import MILVUS_HOST
from config import MILVUS_PORT
from config import VECTOR_DIMENSION
from logs import LOGGER
from pymilvus import Collection
from pymilvus import CollectionSchema
from pymilvus import connections
from pymilvus import DataType
from pymilvus import FieldSchema
from pymilvus import utility


class MilvusHelper:
    """
    the basic operations of PyMilvus

    # This example shows how to:
    #   1. connect to Milvus server
    #   2. create a collection
    #   3. insert entities
    #   4. create index
    #   5. search
    #   6. delete a collection

    """

    def __init__(self):
        try:
            self.collection = None
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            LOGGER.debug(
                f"Successfully connect to Milvus with IP:{MILVUS_HOST} and PORT:{MILVUS_PORT}"
            )
        except Exception as e:
            LOGGER.error(f"Failed to connect Milvus: {e}")
            sys.exit(1)

    def set_collection(self, collection_name):
        try:
            if self.has_collection(collection_name):
                self.collection = Collection(name=collection_name)
            else:
                raise Exception(
                    f"There is no collection named:{collection_name}")
        except Exception as e:
            LOGGER.error(f"Failed to set collection in Milvus: {e}")
            sys.exit(1)

    def has_collection(self, collection_name):
        # Return if Milvus has the collection
        try:
            return utility.has_collection(collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to check state of collection in Milvus: {e}")
            sys.exit(1)

    def create_collection(self, collection_name):
        # Create milvus collection if not exists
        try:
            if not self.has_collection(collection_name):
                field1 = FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    descrition="int64",
                    is_primary=True,
                    auto_id=True)
                field2 = FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    descrition="speaker embeddings",
                    dim=VECTOR_DIMENSION,
                    is_primary=False)
                schema = CollectionSchema(
                    fields=[field1, field2], description="embeddings info")
                self.collection = Collection(
                    name=collection_name, schema=schema)
                LOGGER.debug(f"Create Milvus collection: {collection_name}")
            else:
                self.set_collection(collection_name)
            return "OK"
        except Exception as e:
            LOGGER.error(f"Failed to create collection in Milvus: {e}")
            sys.exit(1)

    def insert(self, collection_name, vectors):
        # Batch insert vectors to milvus collection
        try:
            self.create_collection(collection_name)
            data = [vectors]
            self.set_collection(collection_name)
            mr = self.collection.insert(data)
            ids = mr.primary_keys
            self.collection.load()
            LOGGER.debug(
                f"Insert vectors to Milvus in collection: {collection_name} with {len(vectors)} rows"
            )
            return ids
        except Exception as e:
            LOGGER.error(f"Failed to insert data to Milvus: {e}")
            sys.exit(1)

    def create_index(self, collection_name):
        # Create IVF_FLAT index on milvus collection
        try:
            self.set_collection(collection_name)
            default_index = {
                "index_type": "IVF_SQ8",
                "metric_type": METRIC_TYPE,
                "params": {
                    "nlist": 16384
                }
            }
            status = self.collection.create_index(
                field_name="embedding", index_params=default_index)
            if not status.code:
                LOGGER.debug(
                    f"Successfully create index in collection:{collection_name} with param:{default_index}"
                )
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error(f"Failed to create index: {e}")
            sys.exit(1)

    def delete_collection(self, collection_name):
        # Delete Milvus collection
        try:
            self.set_collection(collection_name)
            self.collection.drop()
            LOGGER.debug("Successfully drop collection!")
            return "ok"
        except Exception as e:
            LOGGER.error(f"Failed to drop collection: {e}")
            sys.exit(1)

    def search_vectors(self, collection_name, vectors, top_k):
        # Search vector in milvus collection
        try:
            self.set_collection(collection_name)
            search_params = {
                "metric_type": METRIC_TYPE,
                "params": {
                    "nprobe": 16
                }
            }
            res = self.collection.search(
                vectors,
                anns_field="embedding",
                param=search_params,
                limit=top_k)
            LOGGER.debug(f"Successfully search in collection: {res}")
            return res
        except Exception as e:
            LOGGER.error(f"Failed to search vectors in Milvus: {e}")
            sys.exit(1)

    def count(self, collection_name):
        # Get the number of milvus collection
        try:
            self.set_collection(collection_name)
            num = self.collection.num_entities
            LOGGER.debug(
                f"Successfully get the num:{num} of the collection:{collection_name}"
            )
            return num
        except Exception as e:
            LOGGER.error(f"Failed to count vectors in Milvus: {e}")
            sys.exit(1)
