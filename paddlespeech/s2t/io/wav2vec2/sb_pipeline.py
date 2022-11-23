import transformers
from hyperpyyaml import load_hyperpyyaml
import dataset
import data_pipeline
from dataloader import make_dataloader
import dataio
import paddle
import tqdm
import numpy
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # Defining tokenizer and loading it
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')

    # 2. Define audio pipeline:
    @data_pipeline.takes("wav")
    @data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = dataio.read_audio(wav)
        return sig

    dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @data_pipeline.takes("transcript")
    @data_pipeline.provides("wrd", "tokens_list", "tokens")
    def text_pipeline(wrd):
        wrd = "".join(wrd.split(" "))
        yield wrd
        tokens_list = tokenizer(wrd)["input_ids"]
        yield tokens_list
        tokens = numpy.array(tokens_list, dtype="int64")
        # tokens = paddle.to_tensor(tokens_list, dtype="int64")
        yield tokens

    dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )




hparams_file = 'train_with_wav2vec.yaml'
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, None)

(
    train_data,
    valid_data,
    test_data,
    tokenizer,
    train_bsampler,
    valid_bsampler,
) = dataio_prepare(hparams)

train_dataloader_opts = hparams["train_dataloader_opts"]
valid_dataloader_opts = hparams["valid_dataloader_opts"]

if train_bsampler is not None:
    train_dataloader_opts = {
        "batch_sampler": train_bsampler,
        "num_workers": hparams["num_workers"],
    }

if valid_bsampler is not None:
    valid_dataloader_opts = {"batch_sampler": valid_bsampler}


train_set = make_dataloader(
    train_data, stage='train', **train_dataloader_opts
)

valid_set = make_dataloader(
    valid_data,
    stage='train',
    **valid_dataloader_opts,
)

# print(len(train_set))

for batch in valid_set:
    print(batch)
print('done')    # exit()