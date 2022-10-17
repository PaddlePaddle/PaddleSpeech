# u2/u2pp Streaming ASR 

## Testing with Aishell Test Data

## Download wav and model

```
run.sh --stop_stage 0
```

### compute feature

```
./run.sh --stage 1 --stop_stage 1
```

### decoding using feature

```
./run.sh --stage 2 --stop_stage 2
```

### decoding using wav


```
./run.sh --stage 3 --stop_stage 3
```
