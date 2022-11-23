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

import numpy
import pymysql
from config import MYSQL_DB
from config import MYSQL_HOST
from config import MYSQL_PORT
from config import MYSQL_PWD
from config import MYSQL_USER
from logs import LOGGER


class MySQLHelper():
    """
    the basic operations of PyMySQL

    # This example shows how to:
    #   1. connect to MySQL server
    #   2. create a table
    #   3. insert data to table
    #   4. search by milvus ids
    #   5. delete table
    """

    def __init__(self):
        self.conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            port=MYSQL_PORT,
            password=MYSQL_PWD,
            database=MYSQL_DB,
            local_infile=True)
        self.cursor = self.conn.cursor()

    def test_connection(self):
        try:
            self.conn.ping()
        except Exception:
            self.conn = pymysql.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                port=MYSQL_PORT,
                password=MYSQL_PWD,
                database=MYSQL_DB,
                local_infile=True)
            self.cursor = self.conn.cursor()

    def create_mysql_table(self, table_name):
        # Create mysql table if not exists
        self.test_connection()
        sql = "create table if not exists " + table_name + "(milvus_id TEXT, audio_path TEXT);"
        try:
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL create table: {table_name} with sql: {sql}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def load_data_to_mysql(self, table_name, data):
        # Batch insert (Milvus_ids, audio_path) to mysql
        self.test_connection()
        sql = "insert into " + table_name + " (milvus_id,audio_path) values (%s,%s);"
        try:
            self.cursor.executemany(sql, data)
            self.conn.commit()
            LOGGER.debug(
                f"MYSQL loads data to table: {table_name} successfully")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def search_by_milvus_ids(self, ids, table_name):
        # Get the audio_path according to the milvus ids
        self.test_connection()
        str_ids = str(ids).replace('[', '').replace(']', '')
        sql = "select audio_path from " + table_name + " where milvus_id in (" + str_ids + ") order by field (milvus_id," + str_ids + ");"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            results = [res[0] for res in results]
            LOGGER.debug("MYSQL search by milvus id.")
            return results
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def delete_table(self, table_name):
        # Delete mysql table if exists
        self.test_connection()
        sql = "drop table if exists " + table_name + ";"
        try:
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL delete table:{table_name}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def delete_all_data(self, table_name):
        # Delete all the data in mysql table
        self.test_connection()
        sql = 'delete from ' + table_name + ';'
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            LOGGER.debug(f"MYSQL delete all data in table:{table_name}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def count_table(self, table_name):
        # Get the number of spk in mysql table
        self.test_connection()
        sql = "select count(spk_id) from " + table_name + ";"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            LOGGER.debug(f"MYSQL count table:{results[0][0]}")
            return results[0][0]
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def create_mysql_table_vpr(self, table_name):
        # Create mysql table if not exists
        self.test_connection()
        sql = "create table if not exists " + table_name + "(spk_id TEXT, audio_path TEXT, embedding TEXT);"
        try:
            self.cursor.execute(sql)
            LOGGER.debug(f"MYSQL create table: {table_name} with sql: {sql}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def load_data_to_mysql_vpr(self, table_name, data):
        # Insert (spk, audio, embedding) to mysql
        self.test_connection()
        sql = "insert into " + table_name + " (spk_id,audio_path,embedding) values (%s,%s,%s);"
        try:
            self.cursor.execute(sql, data)
            LOGGER.debug(
                f"MYSQL loads data to table: {table_name} successfully")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def list_vpr(self, table_name):
        # Get all records in mysql
        self.test_connection()
        sql = "select * from " + table_name + " ;"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            self.conn.commit()
            spk_ids = [res[0] for res in results]
            audio_paths = [res[1] for res in results]
            embeddings = [
                numpy.array(
                    str(res[2]).replace('[', '').replace(']', '').split(","))
                for res in results
            ]
            return spk_ids, audio_paths, embeddings
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def search_audio_vpr(self, table_name, spk_id):
        # Get the audio_path according to the spk_id
        self.test_connection()
        sql = "select audio_path from " + table_name + " where spk_id='" + spk_id + "' ;"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            LOGGER.debug(
                f"MYSQL search by spk id {spk_id} to get audio {results[0][0]}.")
            return results[0][0]
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def delete_data_vpr(self, table_name, spk_id):
        # Delete a record by spk_id in mysql table
        self.test_connection()
        sql = "delete from " + table_name + " where spk_id='" + spk_id + "';"
        try:
            self.cursor.execute(sql)
            LOGGER.debug(
                f"MYSQL delete a record {spk_id} in table {table_name}")
        except Exception as e:
            LOGGER.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)
