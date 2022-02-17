# PaddleSpeech Server

## The environment variables
The path.sh contains the environment variable. 
```bash
source ./bin/path.sh
```

## Add engine_backend on conf/application.yaml
```
engine_backend:
    asr: 'conf/asr/asr.yaml'
    [server name]: [conf yaml file]
```
Currently supporting asr and tts services.

## Start service(command line todo)
```bash
python ./bin/main.py
```

## Client access
Refer to `../tests`


