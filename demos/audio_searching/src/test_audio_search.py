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
from audio_search import app
from fastapi.testclient import TestClient

from utils.utility import download
from utils.utility import unpack

client = TestClient(app)


def download_audio_data():
    """
    Download audio data
    """
    url = "https://paddlespeech.bj.bcebos.com/vector/audio/example_audio.tar.gz"
    md5sum = "52ac69316c1aa1fdef84da7dd2c67b39"
    target_dir = "./"
    filepath = download(url, md5sum, target_dir)
    unpack(filepath, target_dir, True)


def test_drop():
    """
    Delete the collection of Milvus and MySQL
    """
    response = client.post("/audio/drop")
    assert response.status_code == 200


def test_load():
    """
    Insert all the audio files under the file path to Milvus/MySQL
    """
    response = client.post("/audio/load", json={"File": "./example_audio"})
    assert response.status_code == 200
    assert response.json() == {
        'status': True,
        'msg': "Successfully loaded data!"
    }


def test_progress():
    """
    Get the progress of dealing with data
    """
    response = client.get("/progress")
    assert response.status_code == 200
    assert response.json() == "current: 20, total: 20"


def test_count():
    """
    Returns the total number of vectors in the system
    """
    response = client.get("/audio/count")
    assert response.status_code == 200
    assert response.json() == 20


def test_search():
    """
    Search the uploaded audio in Milvus/MySQL
    """
    response = client.post(
        "/audio/search/local?query_audio_path=.%2Fexample_audio%2Ftest.wav")
    assert response.status_code == 200
    assert len(response.json()) == 10


def test_data():
    """
    Get the audio file
    """
    response = client.get("/data?audio_path=.%2Fexample_audio%2Ftest.wav")
    assert response.status_code == 200


if __name__ == "__main__":
    download_audio_data()
    test_load()
    test_count()
    test_search()
    test_drop()
