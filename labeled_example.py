import math
from enum import Enum
from pathlib import Path
from textwrap import wrap
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
from lazy import lazy


class SpectrogramType(Enum):
    power = "power"
    amplitude = "amplitude"
    power_level = "power level"


class LabeledExample:
    def __init__(self, id: str, flac_file: Path, text: str, fourier_window_length: int = 512, hop_length: int = 128,
                 sample_rate: int = 16000):
        # The default values for hop_length and fourier_window_length are powers of 2 near the values specified in the wave2letter paper.
        self.id = id
        self.flac_file = flac_file
        self.text = text
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fourier_window_length = fourier_window_length

    @lazy
    def raw_sound(self) -> (Any, int):
        raw_sound, sample_rate = librosa.load(str(self.flac_file), sr=None)
        assert (sample_rate == self.sample_rate)
        return raw_sound

    @lazy
    def power_spectrogram(self):
        return self.amplitude_spectrogram ** 2

    @lazy
    def amplitude_spectrogram(self):
        return np.abs(self.complex_spectrogram)

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

    def show_spectrogram(self, type: SpectrogramType = SpectrogramType.power_level):
        self.prepare_spectrogram_plot(type)
        plt.show()

    def save_spectrogram(self, target_directory: Path, type: SpectrogramType = SpectrogramType.power_level) -> Path:
        self.prepare_spectrogram_plot(type)
        plt.savefig(str(Path(target_directory, "{}_{}_spectrogram.png".format(self.id, type.value))))

    def highest_detectable_frequency(self):
        return self.sample_rate / 2

    def frequency_step(self):
        return self.sample_rate / self.fourier_window_length

    def duration_in_s(self):
        return self.raw_sound.shape[0] / self.sample_rate

    def spectrogram_by_type(self, type: SpectrogramType):
        if type == SpectrogramType.power:
            return self.power_spectrogram
        if type == SpectrogramType.amplitude:
            return self.amplitude_spectrogram
        if type == SpectrogramType.power_level:
            return self.power_level_spectrogram

        raise ValueError(type)

    def frequency_count(self) -> int:
        return self.amplitude_spectrogram.shape[0]

    def time_step_count(self) -> int:
        return self.amplitude_spectrogram.shape[1]

    def time_step_rate(self):
        return self.time_step_count() / self.duration_in_s()

    def prepare_spectrogram_plot(self, type: SpectrogramType = SpectrogramType.power_level):
        spectrogram = self.spectrogram_by_type(type)
        print(spectrogram.shape, self.raw_sound.shape, self.duration_in_s())
        fig, ax = plt.subplots(1, 1)
        plt.title("\n".join(wrap(type.value + " spectrogram for " + str(self), width=100)))
        plt.xlabel("time / s (time step every {}ms)".format(round(1000 / self.time_step_rate())))
        plt.ylabel("frequency / Hz (level every {}Hz, {} total)".format(self.frequency_count(), self.frequency_step()))
        plt.imshow(
            spectrogram, cmap='gist_heat', origin='lower',
            extent=[0, self.duration_in_s(), 0, self.highest_detectable_frequency()], aspect='auto')

        plt.colorbar(label=type.value + (
            " (only proportional to physical scale)" if type != SpectrogramType.power_level else " / dB (not aligned to a particular base level)"))

        fig.set_size_inches(19.20, 10.80)

    @lazy
    def power_level_spectrogram(self):
        def log_and_trunc(x, min: float = -150) -> float:
            if x == 0:
                return min
            l = 10 * math.log10(x)
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
            str(Path(target_directory,
                     "{}_window{}_hop{}.wav".format(self.id, self.fourier_window_length, self.hop_length))),
            self.reconstructed_sound_from_spectrogram, sr=self.sample_rate)

    def sample_rate_info(self):
        return "time / samples (sample rate {}Hz)".format(self.sample_rate)

    def __str__(self):
        return '{0}: "{1}"'.format(self.id, self.text)
