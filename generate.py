import torch, torchaudio, numpy as np
from diffwave.inference import predict as diffwave_predict


model_dir = '/data/axel/diffwave/wheel'
data_path = "/data/datasets/wheel/wav_22kHz_sliced/Chapter 06 - Surprises_47.wav.spec.npy"
# get your hands on a spectrogram in [N,C,W] format
#x, sr = torchaudio.load(data_path)
spectrogram = np.load(data_path)

audio, sample_rate = diffwave_predict(torch.from_numpy(spectrogram), model_dir, fast_sampling=True)
torchaudio.save("test.wav", audio.cpu(), sample_rate=sample_rate)
