from pathlib import Path

from corpus_provider import CorpusProvider

base_directory = Path(Path.home(), "speechless-data")
corpus = CorpusProvider(base_directory)

example = corpus.examples[0]

example.plot_decibel_spectrogram()
example.plot_reconstructed_sound_from_spectrogram()
example.save_reconstructed_sound_from_spectrogram(base_directory)