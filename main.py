from pathlib import Path

from corpus_provider import CorpusProvider
from labeled_example import LabeledExample
from labeled_example import SpectrogramType

base_directory = Path(Path.home(), "speechless-data")
corpus = CorpusProvider(base_directory)


def save_spectrograms(example: LabeledExample):
    example.save_spectrogram(target_directory=base_directory, type=SpectrogramType.power)
    example.save_spectrogram(target_directory=base_directory, type=SpectrogramType.amplitude)
    example.save_spectrogram(target_directory=base_directory, type=SpectrogramType.decibel)


save_spectrograms(corpus.examples[1])

# example.plot_reconstructed_sound_from_spectrogram()
# example.save_reconstructed_sound_from_spectrogram(base_directory)
