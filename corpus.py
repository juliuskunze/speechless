import random
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy
from numpy import ndarray
from os import makedirs
from typing import List, Iterable, Callable, Tuple

from labeled_example import LabeledExample
from tools import group


class ParsingException(Exception):
    pass


class Corpus:
    __metaclass__ = ABCMeta

    def __init__(self, examples: List[LabeledExample],
                 training_examples: List[LabeledExample],
                 test_examples: List[LabeledExample]):
        self.examples = examples
        self.training_examples = training_examples
        self.test_examples = test_examples

        overlapping = set(e.id for e in self.test_examples).intersection(set(e.id for e in self.training_examples))

        if len(overlapping) > 0:
            raise ValueError("Overlapping training and test set!")

    @abstractmethod
    def csv_rows(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> str:
        raise NotImplementedError

    def summarize_to_csv(self, csv_path: Path) -> None:
        import csv
        with csv_path.open('w', encoding='utf8') as csv_summary_file:
            writer = csv.writer(csv_summary_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for row in self.csv_rows():
                writer.writerow(row)


class CombinedCorpus(Corpus):
    def __init__(self, corpus_providers: List[Corpus]):
        self.corpora = corpus_providers
        super().__init__(
            examples=[example
                      for provider in corpus_providers
                      for example in provider.examples],
            training_examples=[example
                               for provider in corpus_providers
                               for example in provider.training_examples],
            test_examples=[example
                           for provider in corpus_providers
                           for example in provider.test_examples])

    def csv_rows(self) -> List[str]:
        return [row
                for corpus in self.corpora
                for row in corpus.csv_rows()]

    def summary(self) -> str:
        return "\n\n".join([corpus_provider.summary() for corpus_provider in self.corpora]) + \
               "\n\n {} total, {} training, {} test".format(
                   len(self.examples), len(self.training_examples), len(self.test_examples))


class TrainingTestSplit:
    training_only = lambda examples: (examples, [])
    test_only = lambda examples: ([], examples)

    @staticmethod
    def randomly_by_directory(training_share: float = .9) -> Callable[
        [List[LabeledExample]], Tuple[List[LabeledExample], List[LabeledExample]]]:
        def split(examples: List[LabeledExample]) -> Tuple[List[LabeledExample], List[LabeledExample]]:
            examples_by_directory = group(examples, key=lambda e: e.audio_directory)
            directories = examples_by_directory.keys()

            # split must be the same every time:
            random.seed(42)
            training_directories = set(random.sample(directories, int(training_share * len(directories))))

            training_examples = [example for example in examples if example.audio_directory in training_directories]
            test_examples = [example for example in examples if example.audio_directory not in training_directories]

            return training_examples, test_examples

        return split

    @staticmethod
    def randomly(training_share: float = .9) -> Callable[
        [List[LabeledExample]], Tuple[List[LabeledExample], List[LabeledExample]]]:
        def split(examples: List[LabeledExample]) -> Tuple[List[LabeledExample], List[LabeledExample]]:
            # split must be the same every time:
            random.seed(42)
            training_examples = random.sample(examples, int(training_share * len(examples)))
            training_example_set = set(training_examples)
            test_examples = [example for example in examples if example not in training_example_set]

            return training_examples, test_examples

        return split

    @staticmethod
    def overfit(training_example_count: int) -> Callable[
        [List[LabeledExample]], Tuple[List[LabeledExample], List[LabeledExample]]]:
        return lambda examples: (examples[:training_example_count], examples[training_example_count:])

    @staticmethod
    def by_directory(test_directory_name: str = "test") -> Callable[
        [List[LabeledExample]], Tuple[List[LabeledExample], List[LabeledExample]]]:
        def split(examples: List[LabeledExample]) -> Tuple[List[LabeledExample], List[LabeledExample]]:
            training_examples = [example for example in examples if example.audio_directory.name != test_directory_name]
            test_examples = [example for example in examples if example.audio_directory.name == test_directory_name]

            return training_examples, test_examples

        return split


def paginate(sequence: List, page_size: int):
    for start in range(0, len(sequence), page_size):
        yield sequence[start:start + page_size]


class LabeledSpectrogram:
    __metaclass__ = ABCMeta

    @abstractmethod
    def label(self) -> str: raise NotImplementedError

    @abstractmethod
    def spectrogram(self) -> ndarray: raise NotImplementedError


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
    def __init__(self, corpus: Corpus, spectrogram_cache_directory: Path,
                 spectrogram_from_example: Callable[[LabeledExample], ndarray] =
                 lambda x: x.z_normalized_transposed_spectrogram(),
                 batch_size: int = 64):
        # not Path.mkdir() for compatibility with Python 3.4
        makedirs(str(spectrogram_cache_directory), exist_ok=True)

        self.batch_size = batch_size
        self.spectrogram_cache_directory = spectrogram_cache_directory
        self.labeled_training_spectrograms = [
            CachedLabeledSpectrogram(example, spectrogram_cache_directory=spectrogram_cache_directory,
                                     spectrogram_from_example=spectrogram_from_example)
            for example in corpus.training_examples]

        self.labeled_test_spectrograms = [
            CachedLabeledSpectrogram(example, spectrogram_cache_directory=spectrogram_cache_directory,
                                     spectrogram_from_example=spectrogram_from_example)
            for example in corpus.test_examples]

    def preview_batch(self):
        return self.labeled_training_spectrograms[:self.batch_size]

    def training_batches(self) -> Iterable[List[LabeledSpectrogram]]:
        while True:
            yield random.sample(self.labeled_training_spectrograms, self.batch_size)

    def test_batches(self) -> Iterable[List[LabeledSpectrogram]]:
        return paginate(self.labeled_test_spectrograms, self.batch_size)
