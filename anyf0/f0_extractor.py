import dataclasses
import pathlib

import numpy as np
import libf0
import librosa
import parselmouth
import pyworld as pw
import resampy
import torch
import torchcrepe


@dataclasses.dataclass
class F0Extractor:
    wav_path: pathlib.Path
    sampling_rate: int = 44100
    hop_length: int = 512
    f0_min: int = 50
    f0_max: int = 1600
    method: str = "dio"
    x: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.x, self.sampling_rate = librosa.load(self.wav_path, sr=self.sampling_rate)

    @property
    def hop_size(self) -> float:
        return self.hop_length / self.sampling_rate

    @property
    def wav16k(self) -> np.ndarray:
        return resampy.resample(self.x, self.sampling_rate, 16000)

    def extract_f0(self) -> np.ndarray:
        f0 = None
        if self.method == "dio":
            _f0, t = pw.dio(
                self.x.astype("double"),
                self.sampling_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                channels_in_octave=2,
                frame_period=(1000 * self.hop_size),
            )
            f0 = pw.stonemask(self.x.astype("double"), _f0, t, self.sampling_rate)
            f0 = f0.astype("float")
        elif self.method == "harvest":
            f0, _ = pw.harvest(
                self.x.astype("double"),
                self.sampling_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=(1000 * self.hop_size),
            )
            f0 = f0.astype("float")
        elif self.method == "pyin":
            f0, _, _ = librosa.pyin(
                self.wav16k,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=self.sampling_rate,
                hop_length=80,
            )
        elif self.method == "yin":
            f0, _, _ = libf0.yin(
                self.wav16k,
                Fs=16000,
                H=80,
                F_min=self.f0_min,
                F_max=self.f0_max,
            )
        elif self.method == "swipe":
            f0, _, _ = libf0.swipe(
                self.x,
                Fs=self.sampling_rate,
                H=self.hop_length,
                F_min=self.f0_min,
                F_max=self.f0_max,
            )
        elif self.method == "salience":
            f0, _, _ = libf0.salience(
                self.wav16k,
                Fs=16000,
                H=80,
                F_min=self.f0_min,
                F_max=self.f0_max,
            )
        elif self.method == "torchcrepe":
            device = "cuda" if torch.cuda.is_available() else "cpu"

            wav16k_torch = torch.FloatTensor(self.wav16k).unsqueeze(0).to(device)
            f0 = torchcrepe.predict(
                wav16k_torch,
                sample_rate=16000,
                hop_length=80,
                batch_size=1024,
                fmin=self.f0_min,
                fmax=self.f0_max,
                device=device,
            )
            f0 = f0[0].cpu().numpy()
        elif self.method == "parselmouth":
            f0 = (
                parselmouth.Sound(self.x, self.sampling_rate)
                .to_pitch_ac(
                    time_step=self.hop_size,
                    voicing_threshold=0.6,
                    pitch_floor=self.f0_min,
                    pitch_ceiling=self.f0_max,
                )
                .selected_array["frequency"]
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return libf0.hz_to_cents(f0, librosa.midi_to_hz(0))

    def plot_f0(self, f0):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(f0)
        plt.title(self.method)
        plt.xlabel("Time (frames)")
        plt.ylabel("F0 (cents)")
        plt.show()
