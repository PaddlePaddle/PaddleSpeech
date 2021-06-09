import pytest
import librosa
import torch
import matplotlib.pyplot as plt
from scipy.signal import chirp, sweep_poly
from nnAudio.Spectrogram import *
from parameters import *

gpu_idx=0

# librosa example audio for testing
example_y, example_sr = librosa.load(librosa.util.example_audio_file())


@pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)  
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_inverse2(n_fft, hop_length, window, device):
    x = torch.tensor(example_y,device=device)
    stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window).to(device)
    istft = iSTFT(n_fft=n_fft, hop_length=hop_length, window=window).to(device)
    X = stft(x.unsqueeze(0), output_format="Complex")
    x_recon = istft(X, length=x.shape[0], onesided=True).squeeze()
    assert np.allclose(x.cpu(), x_recon.cpu(), rtol=1e-5, atol=1e-3)    

@pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_inverse(n_fft, hop_length, window, device):
    x = torch.tensor(example_y, device=device)
    stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window, iSTFT=True).to(device)
    X = stft(x.unsqueeze(0), output_format="Complex")
    x_recon = stft.inverse(X, length=x.shape[0]).squeeze()
    assert np.allclose(x.cpu(), x_recon.cpu(), rtol=1e-3, atol=1)
    

    
# @pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)

# def test_inverse_GPU(n_fft, hop_length, window):
#     x = torch.tensor(example_y,device=f'cuda:{gpu_idx}')
#     stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window, device=f'cuda:{gpu_idx}')
#     X = stft(x.unsqueeze(0), output_format="Complex")
#     x_recon = stft.inverse(X, num_samples=x.shape[0]).squeeze()
#     assert np.allclose(x.cpu(), x_recon.cpu(), rtol=1e-3, atol=1)


@pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_stft_complex(n_fft, hop_length, window, device):
    x = example_y
    stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Complex")
    X_real, X_imag = X[:, :, :, 0].squeeze(), X[:, :, :, 1].squeeze()
    X_librosa = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window=window)
    real_diff, imag_diff = np.allclose(X_real.cpu(), X_librosa.real, rtol=1e-3, atol=1e-3), \
                            np.allclose(X_imag.cpu(), X_librosa.imag, rtol=1e-3, atol=1e-3)
    
    assert real_diff and imag_diff 
    
# @pytest.mark.parametrize("n_fft, hop_length, window", stft_parameters)    
# def test_stft_complex_GPU(n_fft, hop_length, window):
#     x = example_y
#     stft = STFT(n_fft=n_fft, hop_length=hop_length, window=window, device=f'cuda:{gpu_idx}')
#     X = stft(torch.tensor(x,device=f'cuda:{gpu_idx}').unsqueeze(0), output_format="Complex")
#     X_real, X_imag = X[:, :, :, 0].squeeze().detach().cpu(), X[:, :, :, 1].squeeze().detach().cpu()
#     X_librosa = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window=window)
#     real_diff, imag_diff = np.allclose(X_real, X_librosa.real, rtol=1e-3, atol=1e-3), \
#                             np.allclose(X_imag, X_librosa.imag, rtol=1e-3, atol=1e-3)
    
#     assert real_diff and imag_diff        
    
@pytest.mark.parametrize("n_fft, win_length, hop_length", stft_with_win_parameters) 
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_stft_complex_winlength(n_fft, win_length, hop_length, device):
    x = example_y
    stft = STFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Complex")
    X_real, X_imag = X[:, :, :, 0].squeeze(), X[:, :, :, 1].squeeze()
    X_librosa = librosa.stft(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    real_diff, imag_diff = np.allclose(X_real.cpu(), X_librosa.real, rtol=1e-3, atol=1e-3), \
                            np.allclose(X_imag.cpu(), X_librosa.imag, rtol=1e-3, atol=1e-3)
    assert real_diff and imag_diff    
              
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_stft_magnitude(device):
    x = example_y
    stft = STFT(n_fft=2048, hop_length=512).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Magnitude").squeeze()
    X_librosa, _ = librosa.core.magphase(librosa.stft(x, n_fft=2048, hop_length=512))
    assert np.allclose(X.cpu(), X_librosa, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_stft_phase(device):
    x = example_y
    stft = STFT(n_fft=2048, hop_length=512).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0), output_format="Phase")
    X_real, X_imag = torch.cos(X).squeeze(), torch.sin(X).squeeze()
    _, X_librosa = librosa.core.magphase(librosa.stft(x, n_fft=2048, hop_length=512))

    real_diff, imag_diff = np.mean(np.abs(X_real.cpu().numpy() - X_librosa.real)), \
                            np.mean(np.abs(X_imag.cpu().numpy() - X_librosa.imag))

    # I find that np.allclose is too strict for allowing phase to be similar to librosa.
    # Hence for phase we use average element-wise distance as the test metric.
    assert real_diff < 2e-4 and imag_diff < 2e-4

@pytest.mark.parametrize("n_fft, win_length", mel_win_parameters)  
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_mel_spectrogram(n_fft, win_length, device):
    x = example_y
    melspec = MelSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=512).to(device)
    X = melspec(torch.tensor(x, device=device).unsqueeze(0)).squeeze()
    X_librosa = librosa.feature.melspectrogram(x, n_fft=n_fft, win_length=win_length, hop_length=512)
    assert np.allclose(X.cpu(), X_librosa, rtol=1e-3, atol=1e-3)
    
    
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_cqt_1992(device):
    # Log sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs*t)
    x = chirp(s, f0, 1, f1, method='logarithmic')
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT1992(sr=fs, fmin=220, output_format="Magnitude",
                     n_bins=80, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))


    # Complex
    stft = CQT1992(sr=fs, fmin=220, output_format="Complex",
                     n_bins=80, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))

    # Phase
    stft = CQT1992(sr=fs, fmin=220, output_format="Phase",
                     n_bins=160, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    
    assert True
    
@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_cqt_2010(device):
    # Log sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs*t)
    x = chirp(s, f0, 1, f1, method='logarithmic')
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT2010(sr=fs, fmin=110, output_format="Magnitude",
                     n_bins=160, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))     

    # Complex
    stft = CQT2010(sr=fs, fmin=110, output_format="Complex",
                     n_bins=160, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    
    # Phase
    stft = CQT2010(sr=fs, fmin=110, output_format="Phase",
                     n_bins=160, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))    
    assert True   

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_cqt_1992_v2_log(device):
    # Log sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs*t)
    x = chirp(s, f0, 1, f1, method='logarithmic')
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT1992v2(sr=fs, fmin=55, output_format="Magnitude",
                     n_bins=207, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-1992-mag-ground-truth.npy")
    X = torch.log(X + 1e-5)
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT1992v2(sr=fs, fmin=55, output_format="Complex",
                     n_bins=207, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-1992-complex-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Phase
    stft = CQT1992v2(sr=fs, fmin=55, output_format="Phase",
                     n_bins=207, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-1992-phase-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_cqt_1992_v2_linear(device):
    # Linear sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs*t)
    x = chirp(s, f0, 1, f1, method='linear')
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT1992v2(sr=fs, fmin=55, output_format="Magnitude",
                     n_bins=207, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-1992-mag-ground-truth.npy")
    X = torch.log(X + 1e-5)
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT1992v2(sr=fs, fmin=55, output_format="Complex",
                     n_bins=207, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-1992-complex-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Phase
    stft = CQT1992v2(sr=fs, fmin=55, output_format="Phase",
                     n_bins=207, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-1992-phase-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_cqt_2010_v2_log(device):
    # Log sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs*t)
    x = chirp(s, f0, 1, f1, method='logarithmic')
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT2010v2(sr=fs, fmin=55, output_format="Magnitude",
                     n_bins=207, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))     
    X = torch.log(X + 1e-2)
#     np.save("tests/ground-truths/log-sweep-cqt-2010-mag-ground-truth", X.cpu())
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-2010-mag-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT2010v2(sr=fs, fmin=55, output_format="Complex",
                     n_bins=207, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
#     np.save("tests/ground-truths/log-sweep-cqt-2010-complex-ground-truth", X.cpu())         
    ground_truth = np.load("tests/ground-truths/log-sweep-cqt-2010-complex-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

#     # Phase
#     stft = CQT2010v2(sr=fs, fmin=55, device=device, output_format="Phase",
#                      n_bins=207, bins_per_octave=24)
#     X = stft(torch.tensor(x, device=device).unsqueeze(0))
# #     np.save("tests/ground-truths/log-sweep-cqt-2010-phase-ground-truth", X.cpu())      
#     ground_truth = np.load("tests/ground-truths/log-sweep-cqt-2010-phase-ground-truth.npy")
#     assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_cqt_2010_v2_linear(device):
    # Linear sweep case
    fs = 44100
    t = 1
    f0 = 55
    f1 = 22050
    s = np.linspace(0, t, fs*t)
    x = chirp(s, f0, 1, f1, method='linear')
    x = x.astype(dtype=np.float32)

    # Magnitude
    stft = CQT2010v2(sr=fs, fmin=55, output_format="Magnitude",
                     n_bins=207, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
    X = torch.log(X + 1e-2)
#     np.save("tests/ground-truths/linear-sweep-cqt-2010-mag-ground-truth", X.cpu()) 
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-2010-mag-ground-truth.npy")    
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Complex
    stft = CQT2010v2(sr=fs, fmin=55, output_format="Complex",
                     n_bins=207, bins_per_octave=24).to(device)
    X = stft(torch.tensor(x, device=device).unsqueeze(0))
#     np.save("tests/ground-truths/linear-sweep-cqt-2010-complex-ground-truth", X.cpu())
    ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-2010-complex-ground-truth.npy")
    assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

    # Phase
#     stft = CQT2010v2(sr=fs, fmin=55, device=device, output_format="Phase",
#                      n_bins=207, bins_per_octave=24)
#     X = stft(torch.tensor(x, device=device).unsqueeze(0))
# #     np.save("tests/ground-truths/linear-sweep-cqt-2010-phase-ground-truth", X.cpu())
#     ground_truth = np.load("tests/ground-truths/linear-sweep-cqt-2010-phase-ground-truth.npy")
#     assert np.allclose(X.cpu(), ground_truth, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ['cpu', f'cuda:{gpu_idx}'])
def test_mfcc(device):
    x = example_y
    mfcc = MFCC(sr=example_sr).to(device)
    X = mfcc(torch.tensor(x, device=device).unsqueeze(0)).squeeze()
    X_librosa = librosa.feature.mfcc(x, sr=example_sr)
    assert np.allclose(X.cpu(), X_librosa, rtol=1e-3, atol=1e-3)
    

x = torch.randn((4,44100)) # Create a batch of input for the following Data.Parallel test

@pytest.mark.parametrize("device", [f'cuda:{gpu_idx}'])
def test_STFT_Parallel(device):
    spec_layer = STFT(hop_length=512, n_fft=2048, window='hann', 
                                  freq_scale='no',
                                  output_format='Complex').to(device)
    inverse_spec_layer = iSTFT(hop_length=512, n_fft=2048, window='hann', 
                                  freq_scale='no').to(device)    
    
    spec_layer_parallel = torch.nn.DataParallel(spec_layer)
    inverse_spec_layer_parallel = torch.nn.DataParallel(inverse_spec_layer)
    spec = spec_layer_parallel(x)
    x_recon = inverse_spec_layer_parallel(spec, onesided=True, length=x.shape[-1])
    
    assert np.allclose(x_recon.detach().cpu(), x.detach().cpu(), rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", [f'cuda:{gpu_idx}'])   
def test_MelSpectrogram_Parallel(device):
    spec_layer = MelSpectrogram(sr=22050, n_fft=2048, n_mels=128, hop_length=512,
                                            window='hann', center=True, pad_mode='reflect', 
                                            power=2.0, htk=False, fmin=0.0, fmax=None, norm=1, 
                                            verbose=True).to(device)
    spec_layer_parallel = torch.nn.DataParallel(spec_layer)
    spec = spec_layer_parallel(x)

@pytest.mark.parametrize("device", [f'cuda:{gpu_idx}'])    
def test_MFCC_Parallel(device):
    spec_layer = MFCC().to(device)
    spec_layer_parallel = torch.nn.DataParallel(spec_layer)
    spec = spec_layer_parallel(x)    

@pytest.mark.parametrize("device", [f'cuda:{gpu_idx}'])
def test_CQT1992_Parallel(device):
    spec_layer = CQT1992(fmin=110, n_bins=60, bins_per_octave=12).to(device)
    spec_layer_parallel = torch.nn.DataParallel(spec_layer)
    spec = spec_layer_parallel(x) 

@pytest.mark.parametrize("device", [f'cuda:{gpu_idx}'])    
def test_CQT1992v2_Parallel(device):
    spec_layer = CQT1992v2().to(device)
    spec_layer_parallel = torch.nn.DataParallel(spec_layer)
    spec = spec_layer_parallel(x)        

@pytest.mark.parametrize("device", [f'cuda:{gpu_idx}'])    
def test_CQT2010_Parallel(device):
    spec_layer = CQT2010().to(device)
    spec_layer_parallel = torch.nn.DataParallel(spec_layer)
    spec = spec_layer_parallel(x)  
    
@pytest.mark.parametrize("device", [f'cuda:{gpu_idx}'])    
def test_CQT2010v2_Parallel(device):
    spec_layer = CQT2010v2().to(device)
    spec_layer_parallel = torch.nn.DataParallel(spec_layer)
    spec = spec_layer_parallel(x)       