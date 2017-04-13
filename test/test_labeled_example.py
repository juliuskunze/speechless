from pathlib import Path

import librosa
import numpy as np
from unittest import TestCase

from english_corpus import LibriSpeechCorpus
from labeled_example import SpectrogramType, SpectrogramFrequencyScale

corpus = LibriSpeechCorpus(Path.home() / "speechless-data" / "corpus" / "English", corpus_names=["dev-clean"])


class LabeledExampleTest(TestCase):
    def test(self):
        example = corpus.examples[0]
        mel_power_spectrogram = librosa.feature.melspectrogram(
            y=example.raw_audio, n_fft=example.fourier_window_length, hop_length=example.hop_length,
            sr=example.sample_rate)

        self.assertTrue(np.array_equal(mel_power_spectrogram,
                                       example.z_normalized_transposed_spectrogram(type=SpectrogramType.power,
                                                                                   frequency_scale=SpectrogramFrequencyScale.mel)))
