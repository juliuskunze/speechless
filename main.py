from pathlib import Path

from corpus_provider import CorpusProvider
from labeled_example import LabeledExample
from labeled_example import SpectrogramType

base_directory = Path(Path.home(), "speechless-data")
base_spectrogram_directory = Path(base_directory, "spectrograms")
corpus = CorpusProvider(base_directory)


def save_spectrograms(example: LabeledExample):
    example.save_spectrogram(target_directory=base_spectrogram_directory, type=SpectrogramType.power)
    example.save_spectrogram(target_directory=base_spectrogram_directory, type=SpectrogramType.amplitude)
    example.save_spectrogram(target_directory=base_spectrogram_directory, type=SpectrogramType.power_level)


save_spectrograms(corpus.examples[0])

for i in range(10):
    corpus.examples[i].save_spectrogram(target_directory=base_spectrogram_directory)


# example.plot_reconstructed_sound_from_spectrogram()
# example.save_reconstructed_sound_from_spectrogram(base_directory)
