import random
from os import makedirs
from pathlib import Path
from typing import Callable, List, Iterable

import numpy
from numpy.core.multiarray import ndarray

from labeled_example import LabeledExample
from net import LabeledSpectrogram


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
            spectrogram = self.spectrogram_from_example(self.example)
            numpy.save(str(self.spectrogram_cache_file), spectrogram)
            return spectrogram

        return numpy.load(str(self.spectrogram_cache_file))


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

    def input_size_per_time_step(self) -> int:
        return self.labeled_spectrograms[0].spectrogram().shape[1]

    def test_batch(self):
        return self.labeled_spectrograms[:self.batch_size]

    def training_batches(self) -> Iterable[List[LabeledSpectrogram]]:
        while True:
            yield random.sample(self.labeled_spectrograms, self.batch_size)
