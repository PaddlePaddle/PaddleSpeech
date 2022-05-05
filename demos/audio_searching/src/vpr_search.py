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
import os

import uvicorn
from config import UPLOAD_PATH
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from logs import LOGGER
from mysql_helpers import MySQLHelper
from operations.count import do_count_vpr
from operations.count import do_get
from operations.count import do_list
from operations.drop import do_delete
from operations.drop import do_drop_vpr
from operations.load import do_enroll
from operations.search import do_search_vpr
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

MYSQL_CLI = MySQLHelper()

# Mkdir 'tmp/audio-data'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info(f"Mkdir the path: {UPLOAD_PATH}")


@app.post('/vpr/enroll')
async def vpr_enroll(table_name: str=None,
                     spk_id: str=Form(...),
                     audio: UploadFile=File(...)):
    # Enroll the uploaded audio with spk-id into MySQL
    try:
        if not spk_id:
            return {'status': False, 'msg': "spk_id can not be None"}
        # Save the upload data to server.
        content = await audio.read()
        audio_path = os.path.join(UPLOAD_PATH, audio.filename)
        with open(audio_path, "wb+") as f:
            f.write(content)
        do_enroll(table_name, spk_id, audio_path, MYSQL_CLI)
        LOGGER.info(f"Successfully enrolled {spk_id} online!")
        return {'status': True, 'msg': "Successfully enroll data!"}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}


@app.post('/vpr/enroll/local')
async def vpr_enroll_local(table_name: str=None,
                           spk_id: str=None,
                           audio_path: str=None):
    # Enroll the local audio with spk-id into MySQL
    try:
        do_enroll(table_name, spk_id, audio_path, MYSQL_CLI)
        LOGGER.info(f"Successfully enrolled {spk_id} locally!")
        return {'status': True, 'msg': "Successfully enroll data!"}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/vpr/recog')
async def vpr_recog(request: Request,
                    table_name: str=None,
                    audio: UploadFile=File(...)):
    # Voice print recognition online
    try:
        # Save the upload data to server.
        content = await audio.read()
        query_audio_path = os.path.join(UPLOAD_PATH, audio.filename)
        with open(query_audio_path, "wb+") as f:
            f.write(content)
        host = request.headers['host']
        spk_ids, paths, scores = do_search_vpr(host, table_name,
                                               query_audio_path, MYSQL_CLI)
        for spk_id, path, score in zip(spk_ids, paths, scores):
            LOGGER.info(f"spk {spk_id}, score {score}, audio path {path}, ")
        res = dict(zip(spk_ids, zip(paths, scores)))
        # Sort results by distance metric, closest distances first
        res = sorted(res.items(), key=lambda item: item[1][1], reverse=True)
        LOGGER.info("Successfully speaker recognition online!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/vpr/recog/local')
async def vpr_recog_local(request: Request,
                          table_name: str=None,
                          audio_path: str=None):
    # Voice print recognition locally
    try:
        host = request.headers['host']
        spk_ids, paths, scores = do_search_vpr(host, table_name, audio_path,
                                               MYSQL_CLI)
        for spk_id, path, score in zip(spk_ids, paths, scores):
            LOGGER.info(f"spk {spk_id}, score {score}, audio path {path}, ")
        res = dict(zip(spk_ids, zip(paths, scores)))
        # Sort results by distance metric, closest distances first
        res = sorted(res.items(), key=lambda item: item[1][1], reverse=True)
        LOGGER.info("Successfully speaker recognition locally!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/vpr/del')
async def vpr_del(table_name: str=None, spk_id: dict=None):
    # Delete a record by spk_id in MySQL
    try:
        spk_id = spk_id['spk_id']
        if not spk_id:
            return {'status': False, 'msg': "spk_id can not be None"}
        do_delete(table_name, spk_id, MYSQL_CLI)
        LOGGER.info("Successfully delete a record by spk_id in MySQL")
        return {'status': True, 'msg': "Successfully delete data!"}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.get('/vpr/list')
async def vpr_list(table_name: str=None):
    # Get all records in MySQL
    try:
        spk_ids, audio_paths = do_list(table_name, MYSQL_CLI)
        for i in range(len(spk_ids)):
            LOGGER.debug(f"spk {spk_ids[i]}, audio path {audio_paths[i]}")
        LOGGER.info("Successfully list all records from mysql!")
        return spk_ids, audio_paths
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.get('/vpr/data')
async def vpr_data(
    table_name: str=None,
    spk_id: dict=None, ):
    # Get the audio file from path by spk_id in MySQL
    try:
        spk_id = spk_id['spk_id']
        if not spk_id:
            return {'status': False, 'msg': "spk_id can not be None"}
        audio_path = do_get(table_name, spk_id, MYSQL_CLI)
        LOGGER.info(f"Successfully get audio path {audio_path}!")
        return FileResponse(audio_path)
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.get('/vpr/count')
async def vpr_count(table_name: str=None):
    # Get the total number of spk in MySQL
    try:
        num = do_count_vpr(table_name, MYSQL_CLI)
        LOGGER.info("Successfully count the number of spk!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/vpr/drop')
async def drop_tables(table_name: str=None):
    # Delete the table of MySQL
    try:
        do_drop_vpr(table_name, MYSQL_CLI)
        LOGGER.info("Successfully drop tables in MySQL!")
        return {'status': True, 'msg': "Successfully drop tables!"}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.get('/data')
def audio_path(audio_path):
    # Get the audio file from path
    try:
        LOGGER.info(f"Successfully get audio: {audio_path}")
        return FileResponse(audio_path)
    except Exception as e:
        LOGGER.error(f"get audio error: {e}")
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8002)
