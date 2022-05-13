([简体中文](./README_cn.md)|English)

# Customized Auto Speech Recognition

## introduction
In some cases, we need to recognize the specific sentence with high accuracy. eg: customized keyword spotting, address recognition in navigation apps . customized ASR can slove those issues.

this demo is customized for expense account of taxi, which need to recognize rare address.

## Usage
### 1. Installation
Install docker by runing script setup_docker.sh. And then, install tmux (apt-get install tmux).

### 2. demo
* bash websocket_server.sh.  This script will download resources and libs, and then setup the server.
* In the other terminal of docker, run script websocket_client.sh, the client will send data and get the results.