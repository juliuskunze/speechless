import math
from enum import Enum
from pathlib import Path
from textwrap import wrap

import audioread
import librosa
import os
from lazy import lazy
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from numpy import ndarray, mean, std, vectorize, dot
from typing import List, Callable, Optional

from tools import name_without_extension


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


def z_normalize(array: ndarray) -> ndarray:
    return (array - mean(array)) / std(array)


class LabeledExample:
    def __init__(self, id: str,
                 get_raw_audio: Callable[[], ndarray],
                 label: Optional[str],
                 fourier_window_length: int = 512,
                 hop_length: int = 128,
                 mel_frequency_count: int = 128,
                 sample_rate: int = 16000,
                 original_label_with_tags: Optional[str] = None,
                 get_original_sample_rate: Callable[[], Optional[int]] = lambda: None):
        # The default values for hop_length and fourier_window_length are powers of 2 near the values specified in the wave2letter paper.
        self.sample_rate = sample_rate
        self.id = id
        self._get_raw_audio = get_raw_audio
        self.label = label
        self.fourier_window_length = fourier_window_length
        self.hop_length = hop_length
        self.mel_frequency_count = mel_frequency_count
        self.original_label_with_tags = original_label_with_tags
        self._get_original_sample_rate = get_original_sample_rate

    @staticmethod
    def from_file(audio_file: Path, id: Optional[str] = None,
                  sample_rate_to_convert_to: int = 16000,
                  label_from_id: Callable[[str], Optional[str]] = lambda id: None,
                  fourier_window_length: int = 512,
                  hop_length: int = 128,
                  mel_frequency_count: int = 128,
                  original_label_with_tags_from_id: Callable[[str], Optional[str]] = lambda id: None
                  ) -> 'LabeledExample':
        if id is None:
            id = name_without_extension(audio_file)

        def get_original_sample_rate():
            with audioread.audio_open(os.path.realpath(str(audio_file))) as input_file:
                return input_file.samplerate

        def get_raw_audio():
            y, sample_rate = librosa.load(str(audio_file), sr=sample_rate_to_convert_to)

            return y

        return LabeledExample(id=id, get_raw_audio=get_raw_audio,
                              sample_rate=sample_rate_to_convert_to,
                              label=label_from_id(id),
                              fourier_window_length=fourier_window_length,
                              hop_length=hop_length,
                              mel_frequency_count=mel_frequency_count,
                              get_original_sample_rate=get_original_sample_rate,
                              original_label_with_tags=original_label_with_tags_from_id(id))

    def tag_count(self, tag: str) -> int:
        return self.original_label_with_tags.count(tag)

    @lazy
    def original_sample_rate(self) -> int:
        return self._get_original_sample_rate()

    @lazy
    def raw_audio(self) -> ndarray:
        return self._get_raw_audio()

    def _power_spectrogram(self) -> ndarray:
        return self._amplitude_spectrogram() ** 2

    def _amplitude_spectrogram(self) -> ndarray:
        return abs(self._complex_spectrogram())

    def _complex_spectrogram(self) -> ndarray:
        return librosa.stft(y=self.raw_audio, n_fft=self.fourier_window_length, hop_length=self.hop_length)

    def mel_frequencies(self) -> List[float]:
        # according to librosa.filters.mel code
        return librosa.mel_frequencies(self.mel_frequency_count + 2, fmax=self.sample_rate / 2)

    def _convert_spectrogram_to_mel_scale(self, linear_frequency_spectrogram: ndarray) -> ndarray:
        return dot(
            librosa.filters.mel(sr=self.sample_rate, n_fft=self.fourier_window_length, n_mels=self.mel_frequency_count),
            linear_frequency_spectrogram)

    def plot_raw_audio(self) -> None:
        self._plot_audio(self.raw_audio)

    def _plot_audio(self, audio: ndarray) -> None:
        import matplotlib.pyplot as plt

        plt.title(str(self))
        plt.xlabel("time / samples (sample rate {}Hz)".format(self.sample_rate))
        plt.ylabel("y")
        plt.plot(audio)
        plt.show()

    def show_spectrogram(self, type: SpectrogramType = SpectrogramType.power_level):
        import matplotlib.pyplot as plt

        self.prepare_spectrogram_plot(type)
        plt.show()

    def save_spectrogram(self, target_directory: Path,
                         type: SpectrogramType = SpectrogramType.power_level,
                         frequency_scale: SpectrogramFrequencyScale = SpectrogramFrequencyScale.linear) -> Path:
        import matplotlib.pyplot as plt

        self.prepare_spectrogram_plot(type, frequency_scale)
        path = Path(target_directory, "{}_{}{}_spectrogram.png".format(self.id,
                                                                       "mel_" if frequency_scale == SpectrogramFrequencyScale.mel else "",
                                                                       type.value.replace(" ", "_")))

        plt.savefig(str(path))
        return path

    def highest_detectable_frequency(self) -> float:
        return self.sample_rate / 2

    def duration_in_s(self) -> float:
        return self.raw_audio.shape[0] / self.sample_rate

    def spectrogram(self, type: SpectrogramType = SpectrogramType.power_level,
                    frequency_scale: SpectrogramFrequencyScale = SpectrogramFrequencyScale.linear) -> ndarray:
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

    def z_normalized_transposed_spectrogram(self):
        """
        :return: Array with shape (time, frequencies)
        """
        return z_normalize(self.spectrogram(frequency_scale=SpectrogramFrequencyScale.mel).T)

    def frequency_count_from_spectrogram(self, spectrogram: ndarray) -> int:
        return spectrogram.shape[0]

    def time_step_count(self) -> int:
        return self.spectrogram().shape[1]

    def time_step_rate(self) -> float:
        return self.time_step_count() / self.duration_in_s()

    def prepare_spectrogram_plot(self, type: SpectrogramType = SpectrogramType.power_level,
                                 frequency_scale: SpectrogramFrequencyScale = SpectrogramFrequencyScale.linear) -> None:
        import matplotlib.pyplot as plt

        spectrogram = self.spectrogram(type, frequency_scale=frequency_scale)

        figure, axes = plt.subplots(1, 1)
        use_mel = frequency_scale == SpectrogramFrequencyScale.mel

        plt.title("\n".join(wrap(
            "{0}{1} spectrogram for {2}".format(("mel " if use_mel else ""), type.value, str(self)), width=100)))
        plt.xlabel("time (data every {}ms)".format(round(1000 / self.time_step_rate())))
        plt.ylabel("frequency (data evenly distributed on {} scale, {} total)".format(
            frequency_scale.value, self.frequency_count_from_spectrogram(spectrogram)))
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
            FuncFormatter(lambda value, pos: "{}mel = {}Hz".format(int(value), int(
                librosa.mel_to_hz(value)[0]))) if use_mel else ScalarFormatterWithUnit("Hz"))
        figure.set_size_inches(19.20, 10.80)

    @staticmethod
    def _power_level_from_power_spectrogram(spectrogram: ndarray) -> ndarray:
        # default value for min_decibel found by experiment (all values except for 0s were above this bound)
        def power_to_decibel(x, min_decibel: float = -150) -> float:
            if x == 0:
                return min_decibel
            l = 10 * math.log10(x)
            return min_decibel if l < min_decibel else l

        return vectorize(power_to_decibel)(spectrogram)

    def reconstructed_audio_from_spectrogram(self) -> ndarray:
        return librosa.istft(self._complex_spectrogram(), win_length=self.fourier_window_length,
                             hop_length=self.hop_length)

    def plot_reconstructed_audio_from_spectrogram(self) -> None:
        self._plot_audio(self.reconstructed_audio_from_spectrogram())

    def save_reconstructed_audio_from_spectrogram(self, target_directory: Path) -> None:
        librosa.output.write_wav(
            str(Path(target_directory,
                     "{}_window{}_hop{}.wav".format(self.id, self.fourier_window_length, self.hop_length))),
            self.reconstructed_audio_from_spectrogram(), sr=self.sample_rate)

    def save_spectrograms_of_all_types(self, target_directory: Path) -> None:
        for type in SpectrogramType:
            for frequency_scale in SpectrogramFrequencyScale:
                self.save_spectrogram(target_directory=target_directory, type=type,
                                      frequency_scale=frequency_scale)

    def __str__(self) -> str:
        return self.id + (": {}".format(self.label) if self.label else "")
