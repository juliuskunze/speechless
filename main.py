from pathlib import Path

from corpus_provider import CorpusProvider
from labeled_example import LabeledExample, SpectrogramFrequencyScale
from labeled_example import SpectrogramType

base_directory = Path(Path.home(), "speechless-data")
base_spectrogram_directory = Path(base_directory, "spectrograms")
corpus = CorpusProvider(base_directory)


def save_spectrograms(example: LabeledExample):
    for frequency_scale in SpectrogramFrequencyScale:
            example.save_spectrogram(target_directory=base_spectrogram_directory, frequency_scale=frequency_scale)


save_spectrograms(corpus.examples[0])

#for i in range(10):
#     corpus.examples[i].save_spectrogram(target_directory=base_spectrogram_directory)