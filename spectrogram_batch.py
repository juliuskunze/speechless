import random
from pathlib import Path

import numpy
from numpy.core.multiarray import ndarray
from os import makedirs
from typing import Callable, List, Iterable

from labeled_example import LabeledExample
from net import LabeledSpectrogram


def paginate(sequence: List, page_size: int):
    for start in range(0, len(sequence), page_size):
        yield sequence[start:start + page_size]


class CachedLabeledSpectrogram(LabeledSpectrogram):
    def __init__(self, example: LabeledExample, spectrogram_cache_directory: Path,
                 spectrogram_from_example: Callable[[LabeledExample], ndarray] =
                 lambda x: x.z_normalized_transposed_spectrogram()):
        self.spectrogram_from_example = spectrogram_from_example
        self.example = example
        self.spectrogram_cache_file = spectrogram_cache_directory / "{}.npy".format(example.id)

    def label(self) -> str:
        return self.example.label

    def spectrogram(self) -> ndarray:
        if not self.spectrogram_cache_file.exists():
            return self._calculate_and_save_spectrogram()

        try:
            return numpy.load(str(self.spectrogram_cache_file))
        except ValueError as e:
            print("Recalculating cached file {} because loading failed.".format(self.spectrogram_cache_file))
            return self._calculate_and_save_spectrogram()

    def _calculate_and_save_spectrogram(self):
        spectrogram = self.spectrogram_from_example(self.example)
        numpy.save(str(self.spectrogram_cache_file), spectrogram)
        return spectrogram


class LabeledSpectrogramBatchGenerator:
    def __init__(self, examples: List[LabeledExample], spectrogram_cache_directory: Path,
                 spectrogram_from_example: Callable[[LabeledExample], ndarray] =
                 lambda x: x.z_normalized_transposed_spectrogram(),
                 batch_size: int = 64):
        # not Path.mkdir() for compatibility with Python 3.4
        makedirs(str(spectrogram_cache_directory), exist_ok=True)

        self.batch_size = batch_size
        self.spectrogram_cache_directory = spectrogram_cache_directory
        self.labeled_spectrograms = [
            CachedLabeledSpectrogram(example, spectrogram_cache_directory=spectrogram_cache_directory,
                                     spectrogram_from_example=spectrogram_from_example)
            for example in examples]

    def preview_batch(self):
        return self.labeled_spectrograms[:self.batch_size]

    def as_training_batches(self) -> Iterable[List[LabeledSpectrogram]]:
        while True:
            yield random.sample(self.labeled_spectrograms, self.batch_size)

    def as_validation_batches(self) -> Iterable[List[LabeledSpectrogram]]:
        return paginate(self.labeled_spectrograms, self.batch_size)
