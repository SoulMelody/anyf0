import dataclasses
import pathlib

import libf0
import librosa
import numpy as np
import onnxruntime as ort
import parselmouth
import pyworld as pw

from anyf0.fcpe import FCPE
from anyf0.swift import SWIFT


def hz_to_cents(F, F_ref=55.0):
    """
    Converts frequency in Hz to cents.

    Parameters
    ----------
    F : float or ndarray
        Frequency value in Hz
    F_ref : float
        Reference frequency in Hz (Default value = 55.0)
    Returns
    -------
    F_cents : float or ndarray
        Frequency in cents
    """

    # Avoid division by 0
    F_temp = np.array(F).astype(float)
    F_temp[F_temp == 0] = np.nan

    F_cents = 1200 * np.log2(F_temp / F_ref)

    return F_cents


@dataclasses.dataclass
class F0Extractor:
    wav_path: pathlib.Path
    sample_rate: int = 44100
    hop_length: int = 512
    f0_min: int = 50
    f0_max: int = 1600
    method: str = "praat_ac"
    x: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.x, self.sample_rate = librosa.load(self.wav_path, sr=self.sample_rate)

    @property
    def hop_size(self) -> float:
        return self.hop_length / self.sample_rate

    @property
    def wav16k(self) -> np.ndarray:
        return librosa.resample(self.x, orig_sr=self.sample_rate, target_sr=16000)

    def extract_f0(self) -> np.ndarray:
        f0 = None
        match self.method:
            case "dio":
                _f0, t = pw.dio(
                    self.x.astype("double"),
                    self.sample_rate,
                    f0_floor=self.f0_min,
                    f0_ceil=self.f0_max,
                    channels_in_octave=2,
                    frame_period=(1000 * self.hop_size),
                )
                f0 = pw.stonemask(self.x.astype("double"), _f0, t, self.sample_rate)
                f0 = f0.astype("float")
            case "harvest":
                f0, _ = pw.harvest(
                    self.x.astype("double"),
                    self.sample_rate,
                    f0_floor=self.f0_min,
                    f0_ceil=self.f0_max,
                    frame_period=(1000 * self.hop_size),
                )
                f0 = f0.astype("float")
            case "pyin" | "swipe" | "salience" | "yin":
                f0, _, _ = getattr(libf0, self.method)(
                    self.x,
                    Fs=self.sample_rate,
                    H=self.hop_length,
                    F_min=self.f0_min,
                    F_max=self.f0_max,
                )
            case "piptrack":
                pitches, magnitudes = librosa.piptrack(
                    y=self.wav16k,
                    fmin=self.f0_min,
                    fmax=self.f0_max,
                    sr=16000,
                    hop_length=80,
                )
                max_indexes = np.argmax(magnitudes, axis=0)
                f0 = pitches[max_indexes, range(magnitudes.shape[1])]
            case "crepe_full" | "crepe_tiny":
                pass
            case "fcpe":
                available_providers = ort.get_available_providers()
                ort_providers = ["CPUExecutionProvider"]
                for gpu_provider in ["WebGpuExecutionProvider", "CUDAExecutionProvider"]:
                    if gpu_provider in available_providers:
                        ort_providers.insert(0, gpu_provider)
                model_fcpe = FCPE(
                    "fcpe.onnx",
                    ort_providers=ort_providers,
                    hop_ms=self.hop_size * 1000
                )
                f0, _ = model_fcpe(self.wav16k)
            case "rmvpe":
                pass # TODO rmvpe onnx
            case "swift":
                available_providers = ort.get_available_providers()
                ort_providers = ["CPUExecutionProvider"]
                for gpu_provider in ["WebGpuExecutionProvider", "CUDAExecutionProvider"]:
                    if gpu_provider in available_providers:
                        ort_providers.insert(0, gpu_provider)
                model_swift = SWIFT(
                    sample_rate=16000,
                    hop_size=80,
                    ort_providers=ort_providers,
                )
                f0 = model_swift.get_f0(
                    self.wav16k,
                    f0_min=self.f0_min,
                    f0_max=self.f0_max,
                )
            case "praat_ac" | "praat_cc":
                l_pad = int(np.ceil(1.5 / self.f0_min * self.sample_rate))
                r_pad = int(self.hop_size * ((len(self.x) - 1) // self.hop_size + 1) - len(self.x) + l_pad + 1)
                f0 = (
                    getattr(
                        parselmouth.Sound(np.pad(self.x, (l_pad, r_pad)), self.sample_rate),
                        "to_pitch_" + self.method.rpartition('_')[-1]
                    )(
                        time_step=self.hop_size,
                        voicing_threshold=0.6,
                        pitch_floor=self.f0_min,
                        pitch_ceiling=self.f0_max,
                    )
                    .selected_array["frequency"]
                )
            case _:
                raise ValueError(f"Unknown method: {self.method}")
        return hz_to_cents(f0, librosa.midi_to_hz(0))

    def plot_f0(self, f0):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(f0)
        plt.title(self.method)
        plt.xlabel("Time (frames)")
        plt.ylabel("F0 (cents)")
        plt.show()