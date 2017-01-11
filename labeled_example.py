import math
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
from lazy import lazy


class LabeledExample:
    def __init__(self, id: str, flac_file: Path, text: str, fourier_window_length_in_s: float = 1 / 100,
                 hop_length_in_s: float = 1 / 40, sample_rate: int = 16200):
        self.id = id
        self.flac_file = flac_file
        self.text = text
        self.sample_rate = sample_rate

        def length_in_samples(length_in_seconds: float) -> int:
            return int(length_in_seconds * self.sample_rate)

        self.hop_length = length_in_samples(hop_length_in_s)
        self.fourier_window_length = length_in_samples(fourier_window_length_in_s)

    @lazy
    def raw_sound(self) -> (Any, int):
        raw_sound, sample_rate = librosa.load(str(self.flac_file), sr=None)
        assert (sample_rate == self.sample_rate)
        return raw_sound

    @lazy
    def power_spectrogram(self):
        return np.abs(self.complex_spectrogram) ** 2

    @lazy
    def complex_spectrogram(self):
        return librosa.stft(y=self.raw_sound, n_fft=self.fourier_window_length, hop_length=self.hop_length)

    def plot_raw_sound(self):
        self._plot_sound(self.raw_sound)

    def _plot_sound(self, sound):
        plt.title(str(self))
        plt.xlabel(self.sample_rate_info())
        plt.ylabel("y")
        plt.plot(sound)
        plt.show()

    def plot_decibel_spectrogram(self):
        plt.title(str(self))
        plt.xlabel(self.sample_rate_info())
        plt.ylabel("frequency")
        plt.imshow(self.decibel_spectrogram, cmap='gist_heat')
        plt.colorbar()
        plt.show()

    @lazy
    def decibel_spectrogram(self):
        def log_and_trunc(x, min: float = -15) -> float:
            if x == 0:
                return min
            l = math.log(x)
            return min if l < min else l

        return np.vectorize(log_and_trunc)(self.power_spectrogram)

    @lazy
    def reconstructed_sound_from_spectrogram(self):
        return librosa.istft(self.complex_spectrogram, win_length=self.fourier_window_length,
                             hop_length=self.hop_length)

    def plot_reconstructed_sound_from_spectrogram(self):
        self._plot_sound(self.reconstructed_sound_from_spectrogram)

    def save_reconstructed_sound_from_spectrogram(self, target_directory: Path):
        librosa.output.write_wav(
            str(Path(target_directory, "out_window{}_hop{}.wav".format(self.fourier_window_length, self.hop_length))),
            self.reconstructed_sound_from_spectrogram, sr=self.sample_rate)

    def sample_rate_info(self):
        return "time (sample rate {}Hz)".format(self.sample_rate)

    def __str__(self):
        return '{0}: "{1}"'.format(self.id, self.text)
