([简体中文](./README_cn.md)|English)

# Audio Searching

## Introduction
This demo uses ECAPA-TDNN(or other models) for Speaker Recognition base on MySQL to store user-info/id and Milvus to search vectors.

## Usage
### 1. Prepare MySQL and Milvus services by docker-compose
The molecular similarity search system requires Milvus, MySQL services. We can start these containers with one click through [docker-compose.yaml](./docker-compose.yaml), so please make sure you have [installed Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/) before running. then

```bash
docker-compose -f docker-compose.yaml up -d
```

Then you will see the that all containers are created:

```bash
Creating network "quick_deploy_app_net" with driver "bridge"
Creating milvus-minio    ... done
Creating milvus-etcd     ... done
Creating audio-mysql     ... done
Creating milvus-standalone ... done
```

And show all containers with `docker ps`, and you can use `docker logs audio-mysql` to get the logs of server container

```bash
CONTAINER ID  IMAGE COMMAND CREATED STATUS  PORTS NAMES
b2bcf279e599  milvusdb/milvus:v2.0.1  "/tini -- milvus run…"  22 hours ago  Up 22 hours 0.0.0.0:19530->19530/tcp  milvus-standalone
d8ef4c84e25c  mysql:5.7 "docker-entrypoint.s…"  22 hours ago  Up 22 hours 0.0.0.0:3306->3306/tcp, 33060/tcp audio-mysql
8fb501edb4f3  quay.io/coreos/etcd:v3.5.0  "etcd -advertise-cli…"  22 hours ago  Up 22 hours 2379-2380/tcp milvus-etcd
ffce340b3790  minio/minio:RELEASE.2020-12-03T00-03-10Z  "/usr/bin/docker-ent…"  22 hours ago  Up 22 hours (healthy) 9000/tcp  milvus-minio

```

### 2. Start API Server
Then to start the system server, and it provides HTTP backend services.

- Install the Python packages

```bash
pip install -r requirements.txt
```
- Set configuration

```bash
vim src/config.py
```

Modify the parameters according to your own environment. Here listing some parameters that need to be set, for more information please refer to [config.py](./src/config.py).

| **Parameter**    | **Description**                                       | **Default setting** |
| ---------------- | ----------------------------------------------------- | ------------------- |
| MILVUS_HOST      | The IP address of Milvus, you can get it by ifconfig. If running everything on one machine, most likely 127.0.0.1 | 127.0.0.1           |
| MILVUS_PORT      | Port of Milvus.                                       | 19530               |
| VECTOR_DIMENSION | Dimension of the vectors.                             | 2048                |
| MYSQL_HOST       | The IP address of Mysql.                              | 127.0.0.1           |
| MYSQL_PORT       | Port of Milvus.                                       | 3306                |
| DEFAULT_TABLE    | The milvus and mysql default collection name.         | audio_table          |

- Run the code

Then start the server with Fastapi.

```bash
python src/main.py
```

Then you will see the Application is started:

```bash
INFO:     Started server process [3949]
2022-03-07 17:39:14,864 ｜ INFO ｜ server.py ｜ serve ｜ 75 ｜ Started server process [3949]
INFO:     Waiting for application startup.
2022-03-07 17:39:14,865 ｜ INFO ｜ on.py ｜ startup ｜ 45 ｜ Waiting for application startup.
INFO:     Application startup complete.
2022-03-07 17:39:14,866 ｜ INFO ｜ on.py ｜ startup ｜ 59 ｜ Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8002 (Press CTRL+C to quit)
2022-03-07 17:39:14,867 ｜ INFO ｜ server.py ｜ _log_started_message ｜ 206 ｜ Uvicorn running on http://127.0.0.1:8002 (Press CTRL+C to quit)
```

### 3. Usage

  ```bash
  python ./src/test_main.py
  ```

### 4.Pretrained Models

Here is a list of pretrained models released by PaddleSpeech that can be used by command and python API:

| Model | Sample Rate
| :--- | :---: 
| ecapa_tdnn | 16000
