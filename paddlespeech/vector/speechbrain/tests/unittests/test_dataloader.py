import paddle
import pytest
from paddle.io import TensorDataset
from speechbrain.dataio.dataset import SamplesDataset

def test_saveable_dataloader(tmpdir, device):
    from speechbrain.dataio.dataloader import SaveableDataLoader
    paddle.device.set_device("cpu")
    
    save_file = tmpdir + "/dataloader.ckpt"
    dataset = SamplesDataset(paddle.randn([10, 1]))
    dataloader = SaveableDataLoader(dataset, collate_fn=None)
    data_iterator = iter(dataloader)
    first_item = next(data_iterator)
    assert first_item == dataset[0]
    # Save here:
    dataloader._speechbrain_save(save_file)
    second_item = next(data_iterator)
    assert second_item == dataset[1]
    # Now make a new dataloader and recover:
    new_dataloader = SaveableDataLoader(dataset, collate_fn=None)
    new_dataloader._speechbrain_load(save_file, end_of_epoch=False, device=None)
    new_data_iterator = iter(new_dataloader)
    second_second_item = next(new_data_iterator)
    assert second_second_item == second_item

def test_saveable_dataloader_multiprocess(tmpdir):
    # Same test as above, but with multiprocess dataloading
    from speechbrain.dataio.dataloader import SaveableDataLoader
    paddle.device.set_device("cpu")

    save_file = tmpdir + "/dataloader.ckpt"
    dataset = SamplesDataset(paddle.randn([10, 1]))
    for num_parallel in [1, 2, 3, 4]:
        dataloader = SaveableDataLoader(
            dataset, num_workers=num_parallel, collate_fn=None
        )  # Note num_workers
        data_iterator = iter(dataloader)
        first_item = next(data_iterator)
        assert first_item == dataset[0]
        # Save here, note that this overwrites.
        dataloader._speechbrain_save(save_file)
        second_item = next(data_iterator)
        assert second_item == dataset[1]
        # Cleanup needed for MacOS (open file limit)
        del data_iterator
        del dataloader
        # Now make a new dataloader and recover:
        new_dataloader = SaveableDataLoader(
            dataset, num_workers=num_parallel, collate_fn=None
        )
        new_dataloader._speechbrain_load(
            save_file, end_of_epoch=False, device=None
        )
        new_data_iterator = iter(new_dataloader)
        second_second_item = next(new_data_iterator)
        print("dataset: {}".format(dataset.samples))
        print("second item: {}".format(second_item))
        print("second second item: {}".format(second_second_item))
        assert second_second_item == second_item
        del new_data_iterator
        del new_dataloader

def test_looped_loader(tmpdir):
    # Tests that LoopedLoader will raise StopIteration appropriately
    # And that it can recover and keep the place.
    from speechbrain.dataio.dataloader import LoopedLoader
    paddle.device.set_device("cpu")
    
    save_file = tmpdir + "/loopedloader.ckpt"
    data = range(3)
    dataloader = LoopedLoader(data, epoch_length=2)
    data_iterator = iter(dataloader)
    assert next(data_iterator) == 0
    # Save here, 1 to go:
    dataloader.save(save_file)
    assert next(data_iterator) == 1
    with pytest.raises(StopIteration):
        next(data_iterator)
    # And it can be continued past the range:
    assert next(data_iterator) == 2
    assert next(data_iterator) == 0
    # And again it raises:
    with pytest.raises(StopIteration):
        next(data_iterator)
    # Now make a new dataloader and recover:
    new_dataloader = LoopedLoader(data, epoch_length=2)
    new_dataloader.load(save_file, end_of_epoch=False, device=None)
    new_data_iterator = iter(new_dataloader)
    next(new_data_iterator)
    with pytest.raises(StopIteration):
        next(new_data_iterator)
