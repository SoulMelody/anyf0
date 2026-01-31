# This module is based on code from aqtq314
# https://github.com/aqtq314/VocV1-BreathGen/blob/main/utils/u_rmvpe.py
# These modules are licensed under the MIT License.

import librosa
import numpy as np


class MelSpectrogram:
    def __init__(self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5,):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True,).astype(np.float32)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def __call__(self, audio):
        n_fft = self.n_fft
        win_length = self.win_length
        hop_length = self.hop_length
        fft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann',
            center=True, pad_mode='reflect',)
        magnitude = np.abs(fft)

        mel_output = np.matmul(self.mel_basis, magnitude)
        log_mel_spec = np.log(np.clip(mel_output, a_min=self.clamp, a_max=None))
        return log_mel_spec