# u2/u2pp Streaming ASR 

## introduce

## tutorial 

### how to use in user's project

> Users want to learn how to use the example for their project ! Not just test standard result ! If users want to use these codes in their project, how to use our code ? For example, there are some audios that needs to be recognized, how to get the result . Please show it in README

> What are the requirements for audio format ? 

> if there are some bins or exes builded by speechx（such as `recognizer_main`）, how to find the enterpoint source code ? how to use these program ? What are the meanings of these parameters ? What are the limits of these parameters ? Please show it in README, not just write it in run.sh .


## Testing with Aishell Test Data

### Download wav and model

```
./run.sh --stop_stage 0
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

### Result

> show result in README ! 
