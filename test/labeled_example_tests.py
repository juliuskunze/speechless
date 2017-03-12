from pathlib import Path
from unittest import TestCase

import librosa
import numpy as np

from corpus_provider import CorpusProvider
from labeled_example import SpectrogramType, SpectrogramFrequencyScale

base_directory = Path(Path.home(), "speechless-data")
base_spectrogram_directory = Path(base_directory, "spectrograms")
base_spectrogram_directory.mkdir(exist_ok=True)
corpus = CorpusProvider(base_directory / "corpus", corpus_names=["dev-clean"])


class LabeledExampleTest(TestCase):
    def test(self):
        example = corpus.examples[0]
        mel_power_spectrogram = librosa.feature.melspectrogram(
            y=example.raw_audio, n_fft=example.fourier_window_length, hop_length=example.hop_length,
            sr=example.sample_rate)

        self.assertTrue(np.array_equal(mel_power_spectrogram,
                                       example.spectrogram(type=SpectrogramType.power,
                                                           frequency_scale=SpectrogramFrequencyScale.mel)))