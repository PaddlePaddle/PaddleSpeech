
(简体中文|[English](./README.md))

# 音频相似性检索
## 介绍

随着互联网不断发展，电子邮件、社交媒体照片、直播视频、客服语音等非结构化数据已经变得越来越普遍。如果想要使用计算机来处理这些数据，需要使用 embedding 技术将这些数据转化为向量 vector，然后进行存储、建索引、并查询

但是，当数据量很大，比如上亿条音频要做相似度搜索，就比较困难了。穷举法固然可行，但非常耗时。针对这种场景，该demo 将介绍如何使用开源向量数据库 Milvus 搭建音频相似度检索系统

音频检索（如演讲、音乐、说话人等检索）实现了在海量音频数据中查询并找出相似声音（或相同说话人）片段。音频相似性检索系统可用于识别相似的音效、最大限度减少知识产权侵权等，还可以快速的检索声纹库、帮助企业控制欺诈和身份盗用等。在音频数据的分类和统计分析中，音频检索也发挥着重要作用

在本 demo 中，你将学会如何构建一个音频检索系统，用来检索相似的声音片段。使用基于 PaddleSpeech 预训练模型（音频分类模型，说话人识别模型等）将上传的音频片段转换为向量数据，并存储在 Milvus 中。Milvus 自动为每个向量生成唯一的 ID，然后将 ID 和 相应的音频信息（音频id，音频的说话人id等等）存储在 MySQL，这样就完成建库的工作。用户在检索时，上传测试音频，得到向量，然后在 Milvus 中进行向量相似度搜索，Milvus 返回的检索结果为向量 ID，通过 ID 在 MySQL 内部查询相应的音频信息即可

![音频检索程图](./img/audio_searching.png)

注：该 demo 使用 [CN-Celeb](http://openslr.org/82/) 数据集，包括至少 650000 条音频，3000 个说话人，来建立音频向量库（音频特征，或音频说话人特征），然后通过预设的距离计算方式进行音频（或说话人）检索，这里面数据集也可以使用其他的，根据需要调整，如Librispeech，VoxCeleb，UrbanSound等

## 使用方法
### 1. MySQL 和 Milvus 安装
音频相似度搜索系统需要用到 Milvus, MySQL 服务。 我们可以通过 [docker-compose.yaml](./docker-compose.yaml) 一键启动这些容器，所以请确保在运行之前已经安装了 [Docker Engine](https://docs.docker.com/engine/install/) 和 [Docker Compose](https://docs.docker.com/compose/install/)。 即

```bash
docker-compose -f docker-compose.yaml up -d
```

然后你会看到所有的容器都被创建: 

```bash
Creating network "quick_deploy_app_net" with driver "bridge"
Creating milvus-minio    ... done
Creating milvus-etcd     ... done
Creating audio-mysql     ... done
Creating milvus-standalone ... done
Creating audio-webclient     ... done
```

可以采用'docker ps'来显示所有的容器，还可以使用'docker logs audio-mysql'来获取服务器容器的日志：

```bash
CONTAINER ID  IMAGE COMMAND CREATED STATUS  PORTS NAMES
b2bcf279e599  milvusdb/milvus:v2.0.1  "/tini -- milvus run…"  22 hours ago  Up 22 hours 0.0.0.0:19530->19530/tcp  milvus-standalone
d8ef4c84e25c  mysql:5.7 "docker-entrypoint.s…"  22 hours ago  Up 22 hours 0.0.0.0:3306->3306/tcp, 33060/tcp audio-mysql
8fb501edb4f3  quay.io/coreos/etcd:v3.5.0  "etcd -advertise-cli…"  22 hours ago  Up 22 hours 2379-2380/tcp milvus-etcd
ffce340b3790  minio/minio:RELEASE.2020-12-03T00-03-10Z  "/usr/bin/docker-ent…"  22 hours ago  Up 22 hours (healthy) 9000/tcp  milvus-minio
15c84a506754  iregistry.baidu-int.com/paddlespeech/audio-search-client:1.0  "/bin/bash -c '/usr/…"  22 hours ago  Up 22 hours (healthy) 0.0.0.0:8068->80/tcp  audio-webclient

```

### 2. 配置并启动 API 服务
启动系统服务程序，它会提供基于 Http 后端服务

- 安装服务依赖的 python 基础包

  ```bash
  pip install -r requirements.txt
  ```
- 修改配置

  ```bash
  vim src/config.py
  ```

  请根据实际环境进行修改。 这里列出了一些需要设置的参数，更多信息请参考 [config.py](./src/config.py)  

  | **Parameter**    | **Description**                                       | **Default setting** |
  | ---------------- | ----------------------------------------------------- | ------------------- |
  | MILVUS_HOST      | The IP address of Milvus, you can get it by ifconfig. If running everything on one machine, most likely 127.0.0.1 | 127.0.0.1           |
  | MILVUS_PORT      | Port of Milvus.                                       | 19530               |
  | VECTOR_DIMENSION | Dimension of the vectors.                             | 2048                |
  | MYSQL_HOST       | The IP address of Mysql.                              | 127.0.0.1           |
  | MYSQL_PORT       | Port of Milvus.                                       | 3306                |
  | DEFAULT_TABLE    | The milvus and mysql default collection name.         | audio_table          |

- 运行程序

  启动用 Fastapi 构建的服务

  ```bash
  python src/main.py
  ```

  然后你会看到应用程序启动:

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

### 3. 测试方法
- 准备数据
  ```bash
  wget -c https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz && tar -xvf cn-celeb_v2.tar.gz 
  ```
  注：如果希望快速搭建 demo，可以采用 ./src/test_main.py:download_audio_data 内部的 20 条音频，另外后续结果展示以该集合为例
 
 - 脚本测试（推荐）

    ```bash
    python ./src/test_main.py
    ```
    注：内部将依次下载数据，加载 paddlespeech 模型，提取 embedding，存储建库，检索，删库

    输出：
    ```bash
    Checkpoint path: %your model path%
    Extracting feature from audio No. 1 , 20 audios in total
    Extracting feature from audio No. 2 , 20 audios in total
    ...
    2022-03-09 17:22:13,870 ｜ INFO ｜ main.py ｜ load_audios ｜ 85 ｜ Successfully loaded data, total count: 20
    2022-03-09 17:22:13,898 ｜ INFO ｜ main.py ｜ count_audio ｜ 147 ｜ Successfully count the number of data!
    2022-03-09 17:22:13,918 ｜ INFO ｜ main.py ｜ audio_path ｜ 57 ｜ Successfully load audio: ./example_audio/test.wav
    ...
    2022-03-09 17:22:32,580 ｜ INFO ｜ main.py ｜ search_local_audio ｜ 131 ｜ search result http://testserver/data?audio_path=./example_audio/test.wav, distance 0.0
    2022-03-09 17:22:32,580 ｜ INFO ｜ main.py ｜ search_local_audio ｜ 131 ｜ search result http://testserver/data?audio_path=./example_audio/knife_chopping.wav, distance 0.021805256605148315
    2022-03-09 17:22:32,580 ｜ INFO ｜ main.py ｜ search_local_audio ｜ 131 ｜ search result http://testserver/data?audio_path=./example_audio/knife_cut_into_flesh.wav, distance 0.052762262523174286
    ...
    2022-03-09 17:22:32,582 ｜ INFO ｜ main.py ｜ search_local_audio ｜ 135 ｜ Successfully searched similar audio!
    2022-03-09 17:22:33,658 ｜ INFO ｜ main.py ｜ drop_tables ｜ 159 ｜ Successfully drop tables in Milvus and MySQL!
    ```
  - 前端测试（可选）
  
    在浏览器中输入 127.0.0.1:8068 访问前端页面
    - 上传音频
    
      ![](./img/insert.png)

    - 检索相似音频

      ![](./img/search.png)

### 4. 预训练模型

以下是 PaddleSpeech 提供的预训练模型列表：

| 模型 | 采样率
| :--- | :---: 
| ecapa_tdnn| 16000
| panns_cnn6| 32000
| panns_cnn10| 32000
| panns_cnn14| 32000
