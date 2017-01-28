import math
from enum import Enum
from pathlib import Path
from textwrap import wrap
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
from lazy import lazy
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter


class SpectrogramFrequencyScale(Enum):
    linear = "linear"
    mel = "mel"


class SpectrogramType(Enum):
    power = "power"
    amplitude = "amplitude"
    power_level = "power level"


class ScalarFormatterWithUnit(ScalarFormatter):
    def __init__(self, unit: str):
        super().__init__()
        self.unit = unit

    def __call__(self, x, pos=None) -> str:
        return super().__call__(x, pos) + self.unit


class LabeledExample:
    def __init__(self, id: str, flac_file: Path, label: str, fourier_window_length: int = 512, hop_length: int = 128,
                 sample_rate: int = 16000):
        # The default values for hop_length and fourier_window_length are powers of 2 near the values specified in the wave2letter paper.
        self.id = id
        self.flac_file = flac_file
        self.label = label
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fourier_window_length = fourier_window_length

    @lazy
    def raw_sound(self) -> (Any, int):
        raw_sound, sample_rate = librosa.load(str(self.flac_file), sr=None)
        assert (sample_rate == self.sample_rate)
        return raw_sound

    def _power_spectrogram(self):
        return self._amplitude_spectrogram() ** 2

    def _amplitude_spectrogram(self):
        return np.abs(self._complex_spectrogram())

    def _complex_spectrogram(self):
        return librosa.stft(y=self.raw_sound, n_fft=self.fourier_window_length, hop_length=self.hop_length)

    def mel_frequencies(self):
        # according to librosa.filters.mel
        return librosa.mel_frequencies(128 + 2, fmax=self.sample_rate / 2)

    def _convert_spectrogram_to_mel_scale(self, linear_frequency_spectrogram):
        return np.dot(librosa.filters.mel(sr=self.sample_rate, n_fft=self.fourier_window_length),
                      linear_frequency_spectrogram)

    def plot_raw_sound(self):
        self._plot_sound(self.raw_sound)

    def _plot_sound(self, sound):
        plt.title(str(self))
        plt.xlabel("time / samples (sample rate {}Hz)".format(self.sample_rate))
        plt.ylabel("y")
        plt.plot(sound)
        plt.show()

    def show_spectrogram(self, type: SpectrogramType = SpectrogramType.power_level):
        self.prepare_spectrogram_plot(type)
        plt.show()

    def save_spectrogram(self, target_directory: Path,
                         type: SpectrogramType = SpectrogramType.power_level,
                         frequency_scale: SpectrogramFrequencyScale = SpectrogramFrequencyScale.linear) -> Path:
        self.prepare_spectrogram_plot(type, frequency_scale)
        path = Path(target_directory, "{}_{}{}_spectrogram.png".format(self.id,
                                                                       "mel_" if frequency_scale == SpectrogramFrequencyScale.mel else "",
                                                                       type.value.replace(" ", "_")))

        plt.savefig(str(path))
        return path

    def highest_detectable_frequency(self):
        return self.sample_rate / 2

    def duration_in_s(self):
        return self.raw_sound.shape[0] / self.sample_rate

    def spectrogram(self, type: SpectrogramType = SpectrogramType.power_level,
                    frequency_scale: SpectrogramFrequencyScale = SpectrogramFrequencyScale.linear):
        def spectrogram_by_type():
            if type == SpectrogramType.power:
                return self._power_spectrogram()
            if type == SpectrogramType.amplitude:
                return self._amplitude_spectrogram()
            if type == SpectrogramType.power_level:
                return self._power_level_from_power_spectrogram(self._power_spectrogram())

            raise ValueError(type)

        s = spectrogram_by_type()

        return self._convert_spectrogram_to_mel_scale(s) if frequency_scale == SpectrogramFrequencyScale.mel else s

    @staticmethod
    def frequency_count(spectrogram):
        return spectrogram.shape[0]

    def time_step_count(self) -> int:
        return self.spectrogram().shape[1]

    def time_step_rate(self):
        return self.time_step_count() / self.duration_in_s()

    def prepare_spectrogram_plot(self, type: SpectrogramType = SpectrogramType.power_level,
                                 frequency_scale: SpectrogramFrequencyScale = SpectrogramFrequencyScale.linear):
        spectrogram = self.spectrogram(type, frequency_scale=frequency_scale)

        figure, axes = plt.subplots(1, 1)
        use_mel = frequency_scale == SpectrogramFrequencyScale.mel

        plt.title("\n".join(wrap(
            "{0}{1} spectrogram for {2}".format(("mel " if use_mel else ""), type.value, str(self)), width=100)))
        plt.xlabel("time (data every {}ms)".format(round(1000 / self.time_step_rate())))
        plt.ylabel("frequency (data evenly distributed on {} scale, {} total)".format(frequency_scale.value,
                                                                                      self.frequency_count(
                                                                                          spectrogram)))
        mel_frequencies = self.mel_frequencies()
        plt.imshow(
            spectrogram, cmap='gist_heat', origin='lower', aspect='auto', extent=
            [0, self.duration_in_s(),
             librosa.hz_to_mel(mel_frequencies[0])[0] if use_mel else 0,
             librosa.hz_to_mel(mel_frequencies[-1])[0] if use_mel else self.highest_detectable_frequency()])

        plt.colorbar(label="{} ({})".format(type.value,
                                            "in{} dB, not aligned to a particular base level".format(
                                                " something similar to" if use_mel else "") if type == SpectrogramType.power_level else "only proportional to physical scale"))

        axes.xaxis.set_major_formatter(ScalarFormatterWithUnit("s"))
        axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, pos: "{}mel = {}Hz".format(int(value), int(
                librosa.mel_to_hz(value)[0]))) if use_mel else ScalarFormatterWithUnit("Hz"))
        figure.set_size_inches(19.20, 10.80)

    @staticmethod
    def _power_level_from_power_spectrogram(spectrogram):
        # default value for min_decibel found by experiment (all values except for 0s were above this bound)
        def power_to_decibel(x, min_decibel: float = -150) -> float:
            if x == 0:
                return min_decibel
            l = 10 * math.log10(x)
            return min_decibel if l < min_decibel else l

        return np.vectorize(power_to_decibel)(spectrogram)

    def reconstructed_sound_from_spectrogram(self):
        return librosa.istft(self._complex_spectrogram(), win_length=self.fourier_window_length,
                             hop_length=self.hop_length)

    def plot_reconstructed_sound_from_spectrogram(self):
        self._plot_sound(self.reconstructed_sound_from_spectrogram())

    def save_reconstructed_sound_from_spectrogram(self, target_directory: Path):
        librosa.output.write_wav(
            str(Path(target_directory,
                     "{}_window{}_hop{}.wav".format(self.id, self.fourier_window_length, self.hop_length))),
            self.reconstructed_sound_from_spectrogram(), sr=self.sample_rate)

    def __str__(self):
        return '{0}: "{1}"'.format(self.id, self.label)
