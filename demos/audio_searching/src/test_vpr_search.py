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
from fastapi.testclient import TestClient
from vpr_search import app

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
    Delete the table of MySQL
    """
    response = client.post("/vpr/drop")
    assert response.status_code == 200


def test_enroll_local(spk: str, audio: str):
    """
    Enroll the audio to MySQL
    """
    response = client.post("/vpr/enroll/local?spk_id=" + spk +
                           "&audio_path=.%2Fexample_audio%2F" + audio + ".wav")
    assert response.status_code == 200
    assert response.json() == {
        'status': True,
        'msg': "Successfully enroll data!"
    }


def test_search_local():
    """
    Search the spk in MySQL by audio
    """
    response = client.post(
        "/vpr/recog/local?audio_path=.%2Fexample_audio%2Ftest.wav")
    assert response.status_code == 200


def test_list():
    """
    Get all records in MySQL
    """
    response = client.get("/vpr/list")
    assert response.status_code == 200


def test_data(spk: str):
    """
    Get the audio file by spk_id in MySQL
    """
    response = client.get(
        "/vpr/data",
        json={"spk_id": spk}, )
    assert response.status_code == 200


def test_del(spk: str):
    """
    Delete the record in MySQL by spk_id
    """
    response = client.post(
        "/vpr/del",
        json={"spk_id": spk}, )
    assert response.status_code == 200


def test_count():
    """
    Get the number of spk in MySQL
    """
    response = client.get("/vpr/count")
    assert response.status_code == 200


if __name__ == "__main__":
    download_audio_data()

    test_enroll_local("spk1", "arms_strikes")
    test_enroll_local("spk2", "sword_wielding")
    test_enroll_local("spk3", "test")
    test_list()
    test_data("spk1")
    test_count()
    test_search_local()

    test_del("spk1")
    test_count()
    test_search_local()

    test_enroll_local("spk1", "arms_strikes")
    test_count()
    test_search_local()

    test_drop()
