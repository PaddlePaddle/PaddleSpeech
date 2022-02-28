import paddle
import paddle.nn


def test_BatchNorm1d(device):

    from speechbrain.nnet.normalization import BatchNorm1d

    input = torch.randn(100, 10, ) + 2.0
    norm = BatchNorm1d(input_shape=input.shape).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 20, 10, ) + 2.0
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    # Test with combined dimensions
    input = torch.randn(100, 10, 20, ) + 2.0
    norm = BatchNorm1d(input_shape=input.shape, combine_batch_time=True).to(
        device
    )
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 40, 20, 30, ) + 2.0
    norm = BatchNorm1d(input_shape=input.shape, combine_batch_time=True).to(
        device
    )
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)


def test_BatchNorm2d(device):

    from speechbrain.nnet.normalization import BatchNorm2d

    input = torch.randn(100, 10, 4, 20, ) + 2.0
    norm = BatchNorm2d(input_shape=input.shape).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)


def test_LayerNorm(device):

    from speechbrain.nnet.normalization import LayerNorm

    input = torch.randn(4, 101, 256, ) + 2.0
    norm = LayerNorm(input_shape=input.shape).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=2).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=2).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 101, 16, 32, ) + 2.0
    norm = LayerNorm(input_shape=input.shape).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=[2, 3]).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=[2, 3]).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)


def test_InstanceNorm1d(device):

    from speechbrain.nnet.normalization import InstanceNorm1d

    input = torch.randn(100, 10, 128, ) + 2.0
    norm = InstanceNorm1d(input_shape=input.shape).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=2).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=2).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)


def test_InstanceNorm2d(device):

    from speechbrain.nnet.normalization import InstanceNorm2d

    input = torch.randn(100, 10, 20, 2, ) + 2.0
    norm = InstanceNorm2d(input_shape=input.shape).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=[2, 3]).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=[2, 3]).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)


def test_GroupNorm(device):

    from speechbrain.nnet.normalization import GroupNorm

    input = torch.randn(4, 101, 256, ) + 2.0
    norm = GroupNorm(input_shape=input.shape, num_groups=256).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=2).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=2).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(4, 101, 256, ) + 2.0
    norm = GroupNorm(input_shape=input.shape, num_groups=128).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=2).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=2).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 101, 16, 32, ) + 2.0
    norm = GroupNorm(input_shape=input.shape, num_groups=32).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=3).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=3).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 101, 16, 32, ) + 2.0
    norm = GroupNorm(input_shape=input.shape, num_groups=8).to(device)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=3).mean()
    assert torch.abs(current_mean) < 1e-06

    current_std = output.std(dim=3).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)
