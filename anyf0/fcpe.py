# This module is based on code from aqtq314
# https://github.com/aqtq314/VocV1-BreathGen/blob/main/utils/u_rmvpe.py
# These modules are licensed under the MIT License.
import pathlib
import numpy as np
import onnxruntime as ort

from anyf0.wav2mel import MelSpectrogram


fs16k = 16000

class FCPE:
    def __init__(self, model_path: str, hop_ms=20., ort_providers=['CPUExecutionProvider']):
        model_path = pathlib.Path(model_path)
        self.session = ort.InferenceSession(model_path, providers=ort_providers)

        hop_16k = int(round(fs16k * hop_ms / 1000))
        self.mel_extractor = MelSpectrogram(n_mel_channels=128, sampling_rate=fs16k,
            win_length=1024, hop_length=hop_16k, n_fft=None, mel_fmin=30, mel_fmax=8000)

    def __call__(self, x: np.ndarray):
        x = x.astype(np.float32)

        x0 = x[None]
        xmel = self.mel_extractor(x0)

        n_frames = xmel.shape[-1]
        cents_map = 20 * np.arange(360) + 1997.3794084376191

        xmel = np.pad(xmel, [(0, 0), (0, 0), (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames)], mode='reflect')
        xmel = xmel.transpose(0, 2, 1)
        logits, = self.session.run(None, {
            'mel': xmel,
        })
        logits = logits.squeeze(0)[:n_frames]

        center = np.argmax(logits, axis=1)

        logits_mask = np.abs(np.arange(360) - center[..., None]) <= 4
        logits_masked = logits_mask * logits
        cents_pred = np.sum(logits_masked * cents_map, axis=-1).astype(np.float32) / np.sum(logits_masked, axis=-1).clip(min=1e-9)  # 帧长

        confidence = np.max(logits, axis=-1)

        f0 = 10 * (2 ** (cents_pred / 1200))

        return f0, confidence