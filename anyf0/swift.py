# https://github.com/ArkanDash/Advanced-RVC-Inference/blob/master/advanced_rvc_inference/library/predictors/SWIFT/SWIFT.py

import numpy as np
from swift_f0 import SwiftF0


class SWIFT:
    def __init__(self, sample_rate=16000, hop_size=160, ort_providers=['CPUExecutionProvider']):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.ort_providers = ort_providers

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, confidence_threshold=0.9):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        f0_min = max(f0_min, 46.875)
        f0_max = min(f0_max, 2093.75)

        detector = SwiftF0(
            fmin=f0_min, fmax=f0_max, confidence_threshold=confidence_threshold
        )
        detector.pitch_session.set_providers(self.ort_providers)
        result = detector.detect_from_array(x, self.sample_rate)
        if len(result.timestamps) == 0:
            return np.zeros(p_len)
        target_time = (
            np.arange(p_len) * self.hop_size + self.hop_size / 2
        ) / self.sample_rate
        pitch = np.nan_to_num(result.pitch_hz, nan=0.0)
        pitch[~result.voicing] = 0.0
        f0 = np.interp(target_time, result.timestamps, pitch, left=0.0, right=0.0)

        return f0