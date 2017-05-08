import math
from abc import ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path

import audioread
import librosa
import numpy
import os
from lazy import lazy
from numpy import ndarray, mean, std, vectorize, dot
from typing import List, Optional, Tuple, Callable

from speechless.tools import name_without_extension, mkdir, write_text, log


class SpectrogramFrequencyScale(Enum):
    linear = "linear"
    mel = "mel"


class SpectrogramType(Enum):
    power = "power"
    amplitude = "amplitude"
    power_level = "power level"


def z_normalize(array: ndarray) -> ndarray:
    return (array - mean(array)) / std(array)


class PositionalLabel:
    def __init__(self, labeled_sections: List[Tuple[str, Tuple[float, float]]]):
        if not labeled_sections:
            raise ValueError("Sections must be specified.")

        if any(range is None for label, range in labeled_sections):
            raise ValueError("Range must be specified.")

        self.labeled_sections = labeled_sections
        self.labels = [word for word, range in labeled_sections]
        self.label = " ".join(word for word in self.labels)

    def convert_range_to_seconds(self, original_sample_rate: int) -> 'PositionalLabel':
        return PositionalLabel(list(
            (word, (start_end[0] / original_sample_rate, start_end[1] / original_sample_rate))
            for word, start_end in self.labeled_sections))

    def with_corrected_labels(self, correction: Callable[[str], str]) -> 'PositionalLabel':
        return PositionalLabel([(correction(section), range) for section, range in self.labeled_sections])

    def serialize(self) -> str:
        return "\n".join("{}|{}|{}".format(label, start, end) for label, (start, end) in
                         self.labeled_sections)

    @staticmethod
    def deserialize(serialized: str) -> 'PositionalLabel':
        return PositionalLabel(list((label, (float(start), float(end)))
                                    for label, start, end in
                                    map(lambda item: item.split("|"), serialized.splitlines())))


class LabeledSpectrogram:
    __metaclass__ = ABCMeta

    def __init__(self, id: str, label: str):
        self.label = label
        self.id = id

    @abstractmethod
    def z_normalized_transposed_spectrogram(self) -> ndarray: raise NotImplementedError


class LabeledExample(LabeledSpectrogram):
    def __init__(self,
                 get_raw_audio: Callable[[], ndarray],
                 sample_rate: int = 16000,
                 id: Optional[str] = None,
                 label: Optional[str] = "nolabel",
                 fourier_window_length: int = 512,
                 hop_length: int = 128,
                 mel_frequency_count: int = 128,
                 label_with_tags: str = None,
                 positional_label: Optional[PositionalLabel] = None):
        super().__init__(id=id, label=label)

        # The default values for hop_length and fourier_window_length are powers of 2 near the values specified in the wave2letter paper.
        self.get_raw_audio = get_raw_audio
        self.sample_rate = sample_rate
        self.fourier_window_length = fourier_window_length
        self.hop_length = hop_length
        self.mel_frequency_count = mel_frequency_count
        self.label_with_tags = label_with_tags
        self.positional_label = positional_label

    def tag_count(self, tag: str) -> int:
        return self.label_with_tags.count(tag)

    def _power_spectrogram(self) -> ndarray:
        return self._amplitude_spectrogram() ** 2

    def _amplitude_spectrogram(self) -> ndarray:
        return abs(self._complex_spectrogram())

    def _complex_spectrogram(self) -> ndarray:
        return librosa.stft(y=self.get_raw_audio(), n_fft=self.fourier_window_length, hop_length=self.hop_length)

    def mel_frequencies(self) -> List[float]:
        # according to librosa.filters.mel code
        return librosa.mel_frequencies(self.mel_frequency_count + 2, fmax=self.sample_rate / 2)

    def _convert_spectrogram_to_mel_scale(self, linear_frequency_spectrogram: ndarray) -> ndarray:
        return dot(
            librosa.filters.mel(sr=self.sample_rate, n_fft=self.fourier_window_length, n_mels=self.mel_frequency_count),
            linear_frequency_spectrogram)

    def highest_detectable_frequency(self) -> float:
        return self.sample_rate / 2

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
        return self.time_step_count() / self.duration_in_s

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

    @lazy
    def duration_in_s(self) -> float:
        return len(self.get_raw_audio()) / self.sample_rate

    def __str__(self) -> str:
        return self.id + (": {}".format(self.label) if self.label else "")


class LabeledExampleFromFile(LabeledExample):
    def __init__(self,
                 audio_file: Path,
                 id: Optional[str] = None,
                 sample_rate_to_convert_to: int = 16000,
                 label: Optional[str] = "nolabel",
                 fourier_window_length: int = 512,
                 hop_length: int = 128,
                 mel_frequency_count: int = 128,
                 label_with_tags: str = None,
                 positional_label: Optional[PositionalLabel] = None):
        # The default values for hop_length and fourier_window_length are powers of 2 near the values specified in the wave2letter paper.

        if id is None:
            id = name_without_extension(audio_file)

        self.audio_file = audio_file

        super().__init__(
            id=id, get_raw_audio=lambda: librosa.load(str(self.audio_file), sr=self.sample_rate)[0],
            label=label, sample_rate=sample_rate_to_convert_to,
            fourier_window_length=fourier_window_length, hop_length=hop_length, mel_frequency_count=mel_frequency_count,
            label_with_tags=label_with_tags, positional_label=positional_label)

    @property
    def audio_directory(self):
        return Path(self.audio_file.parent)

    @lazy
    def original_sample_rate(self) -> int:
        return LabeledExampleFromFile.file_sample_rate(self.audio_file)

    @staticmethod
    def file_sample_rate(audio_file: Path) -> int:
        with audioread.audio_open(os.path.realpath(str(audio_file))) as input_file:
            return input_file.samplerate

    @lazy
    def duration_in_s(self) -> float:
        try:
            return librosa.get_duration(filename=str(self.audio_file))
        except Exception as e:
            log("Failed to get duration of {}: {}".format(self.audio_file, e))
            return 0

    def sections(self) -> Optional[List[LabeledExample]]:
        if self.positional_label is None:
            return None

        audio = self.get_raw_audio()

        def section(label, start, end):
            return LabeledExample(
                get_raw_audio=lambda: audio[int(start * self.sample_rate):int(end * self.sample_rate)],
                label=label,
                sample_rate=self.sample_rate,
                fourier_window_length=self.fourier_window_length, hop_length=self.hop_length,
                mel_frequency_count=self.mel_frequency_count)

        return [section(label, start, end) for label, (start, end) in self.positional_label.labeled_sections]


class CachedLabeledSpectrogram(LabeledSpectrogram):
    def __init__(self, original: LabeledSpectrogram, spectrogram_cache_directory: Path):
        super().__init__(id=original.id, label=original.label)
        self.original = original
        self.spectrogram_cache_file = spectrogram_cache_directory / "{}.npy".format(original.id)

    def z_normalized_transposed_spectrogram(self) -> ndarray:
        if not self.is_cached():
            return self._calculate_and_save_spectrogram()

        return self._load_from_cache()

    def _load_from_cache(self):
        try:
            return numpy.load(str(self.spectrogram_cache_file))
        except ValueError:
            log("Recalculating cached file {} because loading failed.".format(self.spectrogram_cache_file))
            return self._calculate_and_save_spectrogram()

    def _calculate_and_save_spectrogram(self):
        spectrogram = self.original.z_normalized_transposed_spectrogram()
        self._save_to_cache(spectrogram)
        return spectrogram

    def _save_to_cache(self, spectrogram: ndarray):
        numpy.save(str(self.spectrogram_cache_file), spectrogram)

    def is_cached(self):
        return self.spectrogram_cache_file.exists()

    def repair_cached_file_if_incorrect(self):
        if not self.is_cached():
            self._calculate_and_save_spectrogram()
            return

        from_cache = self._load_from_cache()
        calculated = self.original.z_normalized_transposed_spectrogram()
        try:
            numpy.testing.assert_almost_equal(calculated, from_cache, decimal=1)
        except AssertionError as e:
            self.move_incorrect_cached_file_to_backup_location_and_save_error(str(e))
            self._save_to_cache(calculated)

    def move_incorrect_cached_file_to_backup_location_and_save_error(self, error_text: str):
        parent_directory = Path(self.spectrogram_cache_file.parent)
        incorrect_cached_backup_directory = Path(parent_directory.parent / (parent_directory.name + "-incorrect"))
        mkdir(incorrect_cached_backup_directory)
        incorrect_backup_file = incorrect_cached_backup_directory / self.spectrogram_cache_file.name
        incorrect_backup_message_file = incorrect_cached_backup_directory / (
            name_without_extension(self.spectrogram_cache_file) + "-error.txt")
        write_text(incorrect_backup_message_file, error_text)
        self.spectrogram_cache_file.rename(incorrect_backup_file)
