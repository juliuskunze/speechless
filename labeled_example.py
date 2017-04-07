import math
from enum import Enum
from pathlib import Path

import audioread
import librosa
import os
from lazy import lazy
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

    def __str__(self) -> str:
        return self.id + (": {}".format(self.label) if self.label else "")
