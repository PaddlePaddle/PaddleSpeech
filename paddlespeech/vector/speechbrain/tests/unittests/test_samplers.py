import paddle
from speechbrain.dataio.dataset import ConcatDataset
from paddle.io import TensorDataset
from paddle.io import DataLoader

def test_ConcatDatasetBatchSampler(device):

    paddle.device.set_device("cpu")
    # from paddle.io import ConcatDataset
    from speechbrain.dataio.sampler import (
        ReproducibleRandomSampler,
        ConcatDatasetBatchSampler,
    )
    import numpy as np

    datasets = []
    for i in range(3):
        if i == 0:
            # note: in paddlepaddle, we need to change the data to list
            datasets.append(
                TensorDataset([paddle.arange(i * 10, (i + 1) * 10)])
            )
        else:
            # note: in paddlepaddle, we need to change the data to list
            datasets.append(
                TensorDataset([paddle.arange(i * 6, (i + 1) * 6)])
            )

    samplers = [ReproducibleRandomSampler(x) for x in datasets]
    dataset = ConcatDataset(datasets)
    loader = DataLoader(
        dataset, batch_sampler=ConcatDatasetBatchSampler(samplers, [1, 1, 1])
    )

    concat_data = []
    for data in loader:
        concat_data.append([x.item() for x in data[0]])
    concat_data = np.array(concat_data)

    non_cat_data = []
    for i in range(len(samplers)):
        c_data = []
        loader = DataLoader(dataset.datasets[i], batch_sampler=samplers[i])

        for data in loader:
            c_data.append(data[0].item())

        non_cat_data.append(c_data)

    minlen = min([len(x) for x in non_cat_data])
    non_cat_data = [x[:minlen] for x in non_cat_data]
    non_cat_data = np.array(non_cat_data)
    np.testing.assert_array_equal(non_cat_data.T, concat_data)