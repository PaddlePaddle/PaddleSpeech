
# Advanced Usage
This section covers how to extend TTS by implementing your models and experiments. Guidelines on implementation are also elaborated.

For the general deep learning experiment, there are several parts to deal with:
1. Preprocess the data according to the needs of the model, and iterate the dataset by batch.
2. Define the model, optimizer, and other components.
3. Write out the training process (generally including forward / backward calculation, parameter update, log recording, visualization, periodic evaluation, etc.).
5. Configure and run the experiment.

## PaddleSpeech TTS's Model Components
To balance the reusability and function of models, we divide models into several types according to their characteristics.

For the commonly used modules that can be used as part of other larger models, we try to implement them as simple and universal as possible, because they will be reused. Modules with trainable parameters are generally implemented as subclasses of `paddle.nn.Layer`. Modules without trainable parameters can be directly implemented as a function, and its input and output are `paddle.Tensor`.

Models for a specific task are implemented as subclasses of `paddle.nn.Layer`. Models could be simple, like a single-layer RNN. For complicated models, it is recommended to split the model into different components.

For a seq-to-seq model, it's natural to split it into encoder and decoder. For a model composed of several similar layers, it's natural to extract the sublayer as a separate layer.

There are two common ways to define a model which consists of several modules.

1. Define a module given the specifications. Here is an example with a multilayer perceptron.
    ```python
    class MLP(nn.Layer):
        def __init__(self, input_size, hidden_size, output_size):
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            return self.linear2(paddle.tanh(self.linear1(x))

    module = MLP(16, 32, 4) # intialize a module
    ```
    When the module is intended to be a generic and reusable layer that can be integrated into a larger model, we prefer to define it in this way.

    For considerations of readability and usability, we strongly recommend **NOT** to pack specifications into a single object. Here’s an example below.
    ```python
    class MLP(nn.Layer):
        def __init__(self, hparams):
            self.linear1 = nn.Linear(hparams.input_size, hparams.hidden_size)
            self.linear2 = nn.Linear(hparams.hidden_size, hparams.output_size)

        def forward(self, x):
            return self.linear2(paddle.tanh(self.linear1(x))
    ```
    For a module defined in this way, it’s harder for the user to initialize an instance. Users have to read the code to check what attributes are used.

    Also, code in this style tends to be abused by passing a huge config object to initialize every module used in an experiment, though each module may not need the whole configuration.

    We prefer to be explicit.

2. Define a module as a combination given its components. Here is an example of a sequence-to-sequence model.
    ```python
    class Seq2Seq(nn.Layer):
        def __init__(self, encoder, decoder):
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            encoder_output = self.encoder(x)
            output = self.decoder(encoder_output)
            return output

    encoder = Encoder(...)
    decoder = Decoder(...)
    # compose two components
    model = Seq2Seq(encoder, decoder)
    ```
    When a model is complicated and made up of several components, each of which has a separate functionality, and can be replaced by other components with the same functionality, we prefer to define it in this way.

In the directory structure of PaddleSpeech TTS, modules with high reusability are placed in `paddlespeech.t2s.modules`, but models for specific tasks are placed in `paddlespeech.t2s.models`. When developing a new model, developers need to consider the feasibility of splitting the modules, and the degree of generality of the modules and place them in appropriate directories.

## PaddleSpeech TTS's Data Components
Another critical component for a deep learning project is data.
PaddleSpeech TTS uses the following methods for training data:
1. Preprocess the data.
2. Load the preprocessed data for training.

Previously, we wrote the preprocessing in the `__getitem__` of the Dataset, which will process when accessing a certain batch sample, but encountered some problems:

1.  Efficiency problem. Even if Paddle has a design to load data asynchronously, when the batch size is large, each sample needs to be preprocessed and set up batches, which takes a lot of time, and may even seriously slow down the training process.
2. Data filtering problem. Some filtering conditions depend on the features of the processed sample. For example, filtering samples that are too short according to text length. If the text length can only be known after `__getitem__`,  every time you filter, the entire dataset needed to be loaded once!  In addition, if you do not pre-filter, A small exception (such as too short text ) in  `__getitem__` will cause an exception in the entire data flow, which is not feasible, because `collate_fn `  presupposes that the acquisition of each sample can be normal. Even if some special flags, such as `None`, are used to mark data acquisition failures, and skip `collate_fn`, it will change batch_size.

Therefore, it is not realistic to put preprocessing entirely on `__getitem__`. We use the method mentioned above instead.
During preprocessing, we can do filtering, We can also save more intermediate features, such as text length, audio length, etc., which can be used for subsequent filtering. Because of the habit of TTS field, data is stored in multiple files, and the processed results are stored in `npy` format.

Use a list-like way to store metadata and store the file path in it, so that you can not be restricted by the specific storage location of the file. In addition to the file path, other metadata can also be stored in it. For example, the path of the text, the path of the audio, the path of the spectrum, the number of frames, the number of sampling points, and so on.

Then for the path, there are multiple opening methods,  such as `sf.read`, `np.load`, etc., so it's best to use a parameter that can be input, we don't even want to determine the reading method by its extension, it's best to let the users input it, in this way, users can define their method to parse the data.

So we learned from the design of `DataFrame`, but our construction method is simpler, only need a `list of dicts`, a dict represents a record, and it's convenient to interact with formats such as `json`, `yaml`. For each selected field, we need to give a parser (called `converter` in the interface), and that's it.

Then we need to select a format for saving metadata to the hard disk. There are two square brackets when storing the list of records in `json`, which is not convenient for stream reading and writing, so we use `jsonlines`. We don't use `yaml` because it occupies too many rows when storing the list of records.

Meanwhile, `cache` is added here, and a multi-process Manager is used to share memory between multiple processes. When `num_workers` is used, it is guaranteed that each sub process will not cache a copy.

The implementation of `DataTable` can be found in `paddlespeech/t2s/datasets/data_table.py`.
```python
class DataTable(Dataset):
    """Dataset to load and convert data for general purpose.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        Metadata, a list of meta datum, each of which is composed of
        several fields
    fields : List[str], optional
        Fields to use, if not specified, all the fields in the data are
        used, by default None
    converters : Dict[str, Callable], optional
        Converters used to process each field, by default None
    use_cache : bool, optional
        Whether to use a cache, by default False

    Raises
    ------
    ValueError
        If there is some field that does not exist in data.
    ValueError
        If there is some field in converters that does not exist in fields.
    """

    def __init__(self,
                 data: List[Dict[str, Any]],
                 fields: List[str]=None,
                 converters: Dict[str, Callable]=None,
                 use_cache: bool=False):
```
Its `__getitem__` method is to parse each field with their parser and then compose a dictionary to return.
```python
def _convert(self, meta_datum: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a meta datum to an example by applying the corresponding
    converters to each field requested.

    Parameters
    ----------
    meta_datum : Dict[str, Any]
        Meta datum

    Returns
    -------
    Dict[str, Any]
        Converted example
    """
    example = {}
    for field in self.fields:
        converter = self.converters.get(field, None)
        meta_datum_field = meta_datum[field]
        if converter is not None:
            converted_field = converter(meta_datum_field)
        else:
            converted_field = meta_datum_field
        example[field] = converted_field
    return example
```

## PaddleSpeech TTS's Training Components
A typical training process includes the following processes:
1. Iterate the dataset.
2. Process batch data.
3. Neural network forward/backward calculation.
4. Parameter update.
5. Evaluate the model on the validation dataset, when some special conditions are reached.
6. Write logs, visualize, and in some cases save necessary intermediate results.
7. Save the state of the model and optimizer.

Here, we mainly introduce the training-related components of TTS in Pa and why we designed it like this.
### Global Reporter
When training and modifying Deep Learning models，logging is often needed, and it has even become the key to model debugging and modifying. We usually use various visualization tools，such as ,  `visualdl` in `paddle`, `tensorboard` in `tensorflow`  and `vidsom`, `wnb` ,etc. Besides, `logging` and `print` are usually used for a different purpose.

In these tools, `print` is the simplest，it doesn't have the concept of  `logger` and `handler` in `logging` 、 `summarywriter`  and `logdir` in `tensorboard`, when printing, there is no need for `global_step` ，It's light enough to appear anywhere in the code, and it's printed to a common stdout. Of course, its customizability is limited, for example, it is no longer intuitive when printing dictionaries or more complex objects. And it's fleeting, people need to use redirection to save information.

For TTS models development，we hope to have a more universal multimedia stdout, which is a tool similar to `tensorboard`, which allows many multimedia forms, but it needs a `summary writer` when using, and a `step` when writing information. If the data are images or voices,  some format control parameters are needed.

This will destroy the modular design to a certain extent. For example, If my model is composed of multiple sublayers, and I want to record some important information in the forward method of some sublayers. For this reason, I may need to pass the `summary writer` to these sublayers, but for the sublayers, its function is the calculation, it should not have extra considerations, and it's also difficult for us to tolerate that the initialization of an `nn.Linear` has an optional `visualizer` in the method. And, for a calculation module, **HOW** can it know the global step? These are things related to the training process!

Therefore, a more common approach is not to put writing_log_code in the definition of layer, but return it, then obtain them during training, and write them to `summary writer`.  However, the return values need to be modified.  `summary writer ` is a broadcaster at the training level, and then each module transmits information to it by modifying the return values.

We think this method is a little ugly. We prefer to return the necessary information only rather than change the return values to accommodate visualization and recording.  When you need to report some information, you should be able to report it without difficulty. So we imitate the design of `chainer` and use the `global repoter`.

It takes advantage of the globality of Python's module-level variables and the effect of context manager.

There is a module-level variable in  `paddlespeech/t2s/training/reporter.py`  `OBSERVATIONS`，which is a `Dict` to store key-value.
```python
# paddlespeech/t2s/training/reporter.py

@contextlib.contextmanager
def scope(observations):
    # make `observation` the target to report to.
    # it is basically a dictionary that stores temporary observations
    global OBSERVATIONS
    old = OBSERVATIONS
    OBSERVATIONS = observations

    try:
        yield
    finally:
        OBSERVATIONS = old
```

Then we implement a context manager `scope`, which is used to switch the variables bound by the name of `OBSERVATIONS`. Then a `getter` function is defined to get the dictionary bound by `OBSERVATIONS`.
```python
def get_observations():
    global OBSERVATIONS
    return OBSERVATIONS
```
Then we define a function to get  the current `OBSERVATIONS`，and write key-value pair into it.
```python
def report(name, value):
    # a simple function to report named value
    # you can use it everywhere, it will get the default target and writ to it
    # you can think of it as std.out
    observations = get_observations()
    if observations is None:
        return
    else:
        observations[name] = value
```
The test code following shows the usage method.
- use `first` as the current `OBSERVATION`, write `first_begin=1`,
- then, open the second `OBSERVATION`, write `second_begin=2`,
- then, open the third `OBSERVATION`, write  `third_begin=3`
- exit the third `OBSERVATION` , we back to the second  `OBSERVATION` automatically
- write some context in the second `OBSERVATION` , then exit it, and   we back to the first  `OBSERVATION` automatically
```python
def test_reporter_scope():
    first = {}
    second = {}
    third = {}

    with scope(first):
        report("first_begin", 1)
        with scope(second):
            report("second_begin", 2)
            with scope(third):
                report("third_begin", 3)
                report("third_end", 4)
            report("seconf_end", 5)
        report("first_end", 6)

    assert first == {'first_begin': 1, 'first_end': 6}
    assert second == {'second_begin': 2, 'seconf_end': 5}
    assert third == {'third_begin': 3, 'third_end': 4}
```

In this way, when we write modular components, we can directly call `report`.  The caller will decide where to report as long as it's ready for `OBSERVATION`, then it opens a `scope` and calls the component within this `scope`.

 The `Trainer` in PaddleSpeech TTS report the information in this way.
```python
while True:
    self.observation = {}
    # set observation as the report target
    # you can use report freely in Updater.update()

    # updating parameters and state
    with scope(self.observation):
        update() # training for a step is defined here
```
### Updater: Model Training Process

To maintain the purity of function and the reusability of code, we abstract the model code into a subclass of  `paddle.nn.Layer`, and write the core computing functions in it.

We tend to write the forward process of training in `forward()`, but only write to the prediction result, not to the loss. Therefore, this module can be called by a larger module.

However, when we compose an experiment, we need to add some other things, such as the training process, evaluation process, checkpoint saving, visualization, and the like. In this process, we will encounter some things that only exist in the training process, such as `optimizer`, `learning rate scheduler`, `visualizer`, etc. These things are not part of the model, they should **NOT** be written in the model code.

We made an abstraction for these intermediate processes, that is, `Updater`, which takes the `model`, `optimizer`, and `data stream` as input, and its function is training. Since there may be differences in training methods of different models, we tend to write a corresponding `Updater` for each model. But this is different from the final training script, there is still a certain degree of encapsulation, just to extract the details of regular saving, visualization, evaluation, etc., and only retain the most basic function, that is,  training the model.

### Visualizer
Because we choose observation as the communication mode, we can simply write the things in observation into `visualizer`.

## PaddleSpeech TTS's Configuration Components
Deep learning experiments often have many options to configure. These configurations can be roughly divided into several categories.
1. Data source and data processing mode configuration.
2. Save path configuration of experimental results.
3. Data preprocessing mode configuration.
4. Model structure and hyperparameter configuration.
5. Training process configuration.

It’s common to change the running configuration to compare results. To keep track of running configuration, we use `yaml` configuration files.

Also, we want to interact with command-line options. Some options that usually change according to running environments are provided by command line arguments. In addition, we want to override an option in the config file without editing it.

Taking these requirements into consideration, we use [yacs](https://github.com/rbgirshick/yacs) as a config management tool. Other tools like [omegaconf](https://github.com/omry/omegaconf) are also powerful and have similar functions.

In each example provided, there is a `config.py`,  the default config is defined at `conf/default.yaml`. If you want to get the default config, import `config.py` and call `get_cfg_defaults()` to get it. Then it can be updated with `yaml` config file or command-line arguments if needed.

For details about how to use yacs in experiments, see [yacs](https://github.com/rbgirshick/yacs).

The following is the basic  `ArgumentParser`:
1. `--config`  is used to support configuration file parsing, and the configuration file itself handles the unique options of each experiment.
2. `--train-metadata` is the path to the training data.
3.  `--output-dir` is the dir to save the training results.（if there are checkpoints in  `checkpoints/` of  `--output-dir` , it defaults to reload the newest checkpoint to train)
4. `--ngpu` determine operation modes，`--ngpu` refers to the number of training processes. If `ngpu` > 0, it means using GPU, else CPU is used.

Developers can refer to the examples in `examples` to write the default configuration file when adding new experiments.

## PaddleSpeech TTS's Experiment template

The experimental codes in PaddleSpeech TTS are generally organized as follows:

```text
.
├──  README.md               (help information)
├──  conf
│     └── default.yaml       (defalut config)
├──  local
│    ├──  preprocess.sh      (script to call data preprocessing.py)
│    ├──  synthesize.sh      (script to call synthesis.py)  
│    ├──  synthesize_e2e.sh  (script to call synthesis_e2e.py)
│    └──train.sh             (script to call train.py)
├── path.sh                  (script include paths to be sourced)
└── run.sh                   (script to call scripts in local)
```
The `*.py` files called by above `*.sh` are located `${BIN_DIR}/`

We add a named argument. `--output-dir` to each training script to specify the output directory. The directory structure is as follows, developers should follow this specification:
```text
exp/default/
├── checkpoints/
│   ├── records.jsonl        (record file)
│   └── snapshot_iter_*.pdz  (checkpoint files)
├── config.yaml              (config file of this experiment)
├── vdlrecords.*.log         (visualdl record file)
├── worker_*.log             (text logging, one file per process)
├── validation/              (output dir during training, information_iter_*/ is the output of each step, if necessary)
├── inference/               (output dir of exported static graph model, which is only used in the final stage of training, if implemented)
└── test/                    (output dir of synthesis results)
```

You can view the examples we provide in `examples`. These experiments are provided to users as examples that can be run directly. Users are welcome to add new models and experiments and contribute code to PaddleSpeech.
