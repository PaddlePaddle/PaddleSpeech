# SpeechX -- Speech Inference All in One

> Test under `Ubuntu 16.04.7 LTS`.

## Build

```
./build.sh
```l

## Valgrind

> If using docker please check `--privileged` is set when `docker run`.

1. Fatal error at startup: a function redirection which is mandatory for this platform-tool combination cannot be set up
```
apt-get install libc6-dbg
```

```
pushd tools
./setup_valgrind.sh
popd
```
