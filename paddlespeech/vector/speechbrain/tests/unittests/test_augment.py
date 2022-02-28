import os
import paddle

import numpy as np

import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, BatchSampler, DataLoader
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import ReproducibleRandomSampler
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataio import write_audio
from speechbrain.dataio.legacy import ExtendedCSVDataset
def test_add_noise(tmpdir, device):
    paddle.device.set_device(device)
    from speechbrain.processing.speech_augmentation import AddNoise

    # Test concatenation of batches
    # 这里默认的竟然不是float32类型
    wav_a = paddle.sin(paddle.arange(8000.0, dtype="float32")).unsqueeze(0)
    a_len = paddle.ones([1], dtype="float32")
    wav_b = (
        paddle.cos(paddle.arange(10000.0, dtype="float32"))
        .unsqueeze(0)
        .tile([2, 1])
    )
    b_len = paddle.ones([2], dtype="float32")
    concat, lens = AddNoise._concat_batch(wav_a, a_len, wav_b, b_len)
    assert concat.shape == [3, 10000]
    assert lens.allclose(paddle.to_tensor([0.8, 1, 1]))
    concat, lens = AddNoise._concat_batch(wav_b, b_len, wav_a, a_len)
    assert concat.shape == [3, 10000]
    expected = paddle.to_tensor([1, 1, 0.8])
    assert lens.allclose(expected)

    test_waveform = paddle.sin(paddle.arange(16000.0, dtype="float32")).unsqueeze(0)
    test_noise = paddle.cos(paddle.arange(16000.0,  dtype="float32")).unsqueeze(0)
    wav_lens = paddle.ones([1],  dtype="float32")

    # Put noise waveform into temporary file
    noisefile = os.path.join(tmpdir, "noise.wav")
    write_audio(noisefile, test_noise.transpose(perm=[1, 0]), 16000)

    csv = os.path.join(tmpdir, "noise.csv")
    with open(csv, "w") as w:
        w.write("ID, duration, wav, wav_format, wav_opts\n")
        w.write(f"1, 1.0, {noisefile}, wav,\n")

    # Edge cases
    no_noise = AddNoise(mix_prob=0.0)
    assert no_noise(test_waveform, wav_lens).allclose(test_waveform)
    no_noise = AddNoise(snr_low=1000, snr_high=1000)
    # assert no_noise(test_waveform, wav_lens).allclose(test_waveform)
    all_noise = AddNoise(csv_file=csv, snr_low=-1000, snr_high=-1000)
    noise_waveform = all_noise(test_waveform, wav_lens)
    assert all_noise(test_waveform, wav_lens).allclose(test_noise, atol=1e-4)
    # Basic 0dB case
    add_noise = AddNoise(csv_file=csv)
    expected = (test_waveform + test_noise) / 2
    assert add_noise(test_waveform, wav_lens).allclose(expected, atol=1e-4)


def test_add_reverb(tmpdir, device):
    paddle.device.set_device(device)
    from speechbrain.processing.speech_augmentation import AddReverb

    test_waveform = paddle.sin(paddle.arange(16000.0, dtype="float32")).unsqueeze(0)
    impulse_response = paddle.zeros([1, 8000], dtype="float32")
    impulse_response[0, 0] = 1.0
    wav_lens = paddle.ones([1], dtype="float32")

    # Put ir waveform into temporary file
    ir1 = os.path.join(tmpdir, "ir1.wav")
    ir2 = os.path.join(tmpdir, "ir2.wav")
    ir3 = os.path.join(tmpdir, "ir3.wav")
    write_audio(ir1, impulse_response.transpose(perm=[0, 1]), 16000)

    impulse_response[0, 0] = 0.0
    impulse_response[0, 10] = 0.5
    write_audio(ir2, impulse_response.transpose(perm=[0, 1]), 16000)

    # Check a very simple non-impulse-response case:
    impulse_response[0, 10] = 0.6
    impulse_response[0, 11] = 0.4
    # sf.write(ir3, impulse_response.squeeze(0).numpy(), 16000)
    write_audio(ir3, impulse_response.transpose(perm=[0, 1]), 16000)
    ir3_result = test_waveform * 0.6 + test_waveform.roll(1, -1) * 0.4

    # write ir csv file
    csv = os.path.join(tmpdir, "ir.csv")
    with open(csv, "w") as w:
        w.write("ID, duration, wav, wav_format, wav_opts\n")
        w.write(f"1, 0.5, {ir1}, wav,\n")
        w.write(f"2, 0.5, {ir2}, wav,\n")
        w.write(f"3, 0.5, {ir3}, wav,\n")

    # Edge case
    no_reverb = AddReverb(csv, reverb_prob=0.0)
    assert no_reverb(test_waveform, wav_lens).allclose(test_waveform)

    # Normal cases
    add_reverb = AddReverb(csv, sorting="original")
    reverbed = add_reverb(test_waveform, wav_lens)[:, 0:1000]
    assert reverbed.allclose(test_waveform[:, 0:1000], atol=1e-1)
    reverbed = add_reverb(test_waveform, wav_lens)[:, 0:1000]
    assert reverbed.allclose(test_waveform[:, 0:1000], atol=1e-1)
    reverbed = add_reverb(test_waveform, wav_lens)[:, 0:1000]
    assert reverbed.allclose(ir3_result[:, 0:1000], atol=2e-1)


def test_speed_perturb(device):
    paddle.device.set_device(device)
    from speechbrain.processing.speech_augmentation import SpeedPerturb

    test_waveform = paddle.sin(paddle.arange(16000.0, dtype="float32")).unsqueeze(0)

    # Edge cases
    no_perturb = SpeedPerturb(16000, perturb_prob=0.0)
    assert no_perturb(test_waveform).allclose(test_waveform)
    no_perturb = SpeedPerturb(16000, speeds=[100])
    assert no_perturb(test_waveform).allclose(test_waveform)

    # # Half speed
    half_speed = SpeedPerturb(16000, speeds=[50])
    assert half_speed(test_waveform).allclose(test_waveform[:, ::2], atol=3e-1)


def test_babble(device):
    paddle.device.set_device(device)
    from speechbrain.processing.speech_augmentation import AddBabble

    test_waveform = paddle.stack(
        (
            paddle.sin(paddle.arange(16000.0, dtype="float32")),
            paddle.cos(paddle.arange(16000.0, dtype="float32")),
        )
    )
    lengths = paddle.ones([2])

    # Edge cases
    no_babble = AddBabble(mix_prob=0.0)
    assert no_babble(test_waveform, lengths).allclose(test_waveform)
    no_babble = AddBabble(speaker_count=1, snr_low=1000, snr_high=1000)
    assert no_babble(test_waveform, lengths).allclose(test_waveform)

    # One babbler just averages the two speakers
    babble = AddBabble(speaker_count=1).to(device)
    expected = (test_waveform + test_waveform.roll(1, 0)) / 2
    assert babble(test_waveform, lengths).allclose(expected, atol=1e-4)


def test_drop_freq(device):
    paddle.device.set_device(device)
    from speechbrain.processing.speech_augmentation import DropFreq

    test_waveform = paddle.sin(paddle.arange(16000.0, dtype="float32")).unsqueeze(0)

    # Edge cases
    no_drop = DropFreq(drop_prob=0.0)
    assert no_drop(test_waveform).allclose(test_waveform)
    no_drop = DropFreq(drop_count_low=0, drop_count_high=0)
    assert no_drop(test_waveform).allclose(test_waveform)

    # Check case where frequency range *does not* include signal frequency
    drop_diff_freq = DropFreq(drop_freq_low=0.5, drop_freq_high=0.9)
    assert drop_diff_freq(test_waveform).allclose(test_waveform, atol=1e-1)

    # Check case where frequency range *does* include signal frequency
    drop_same_freq = DropFreq(drop_freq_low=0.28, drop_freq_high=0.28)
    assert drop_same_freq(test_waveform).allclose(
        paddle.zeros([1, 16000]), atol=4e-1
    )


def test_drop_chunk(device):
    paddle.device.set_device(device)
    from speechbrain.processing.speech_augmentation import DropChunk

    test_waveform = paddle.sin(paddle.arange(16000.0, dtype="float32")).unsqueeze(0)
    lengths = paddle.ones([1])

    # Edge cases
    no_drop = DropChunk(drop_prob=0.0)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)
    no_drop = DropChunk(drop_length_low=0, drop_length_high=0)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)
    no_drop = DropChunk(drop_count_low=0, drop_count_high=0)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)
    no_drop = DropChunk(drop_start=0, drop_end=0)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)

    # Specify all parameters to ensure it is deterministic
    dropper = DropChunk(
        drop_length_low=100,
        drop_length_high=100,
        drop_count_low=1,
        drop_count_high=1,
        drop_start=100,
        drop_end=200,
        noise_factor=0.0,
    )
    expected_waveform = test_waveform.clone()
    expected_waveform[:, 100:200] = 0.0

    assert dropper(test_waveform, lengths).allclose(expected_waveform)

    # Make sure amplitude is similar before and after
    dropper = DropChunk(noise_factor=1.0)
    drop_amplitude = dropper(test_waveform, lengths).abs().mean()
    orig_amplitude = test_waveform.abs().mean()
    assert drop_amplitude.allclose(orig_amplitude, atol=1e-2)


def test_clip(device):
    paddle.device.set_device(device)
    from speechbrain.processing.speech_augmentation import DoClip

    test_waveform = paddle.sin(paddle.arange(16000.0, dtype="float32")).unsqueeze(0)

    # Edge cases
    no_clip = DoClip(clip_prob=0.0)
    assert no_clip(test_waveform).allclose(test_waveform)
    no_clip = DoClip(clip_low=1, clip_high=1)
    assert no_clip(test_waveform).allclose(test_waveform)

    # Sort of a reimplementation of clipping, but its one function call.
    expected = test_waveform.clip(min=-0.5, max=0.5)
    half_clip = DoClip(clip_low=0.5, clip_high=0.5)
    assert half_clip(test_waveform).allclose(expected)


# if __name__ == "__main__":
#     # 调试的过程中进入到了gpu的模式
#     # paddle.disable_static()
#     paddle.device.set_device("cpu")
#     # test_add_noise("./", "cpu")

#     BATCH_NUM = 20
#     BATCH_SIZE = 1

#     IMAGE_SIZE = 10
#     CLASS_NUM = 2

#     USE_GPU = False # whether use GPU to run model

#     # define a random dataset
#     class RandomDataset(Dataset):
#         def __init__(self, num_samples):
#             self.num_samples = num_samples

#         def __getitem__(self, idx):
#             image = np.random.random([IMAGE_SIZE]).astype('float32')
#             label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
#             return image, label

#         def __len__(self):
#             return self.num_samples

#     dataset = ExtendedCSVDataset("./noise.csv")
#     # loader kwargs: {'batch_size': 1, 'num_workers': 0, 'collate_fn': <class 'speechbrain.dataio.batch.PaddedBatch'>, 'batch_sampler': <speechbrain.dataio.sampler.ReproducibleRandomSampler object at 0x7fa8a15afb50>}
#     batch_sampler = ReproducibleRandomSampler(dataset, 
#                                 batch_size=1, 
#                                 shuffle=True,                         
#                                 drop_last=True,
#                                 )
#     loader_kwargs = {}
#     loader_kwargs["batch_sampler"] = batch_sampler
#     loader_kwargs["batch_size"] = 1
#     loader_kwargs["num_workers"] = 0
#     from speechbrain.dataio.batch import padded_batch_collate
#     loader_kwargs["collate_fn"] = padded_batch_collate
#     loader = SaveableDataLoader(dataset, 
#                                 **loader_kwargs)
#     # loader = SaveableDataLoader(dataset, 
#     #                             batch_sampler=batch_sampler, 
#     #                             num_workers=0)

#     data_iter = iter(loader)
#     print("data: {}".format(next(data_iter)))