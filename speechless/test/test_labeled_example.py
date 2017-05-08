import librosa
import numpy as np
from unittest import TestCase

from speechless.english_corpus import LibriSpeechCorpus
from speechless.labeled_example import SpectrogramType, SpectrogramFrequencyScale, PositionalLabel
from speechless.tools import home_directory

corpus = LibriSpeechCorpus(home_directory() / "speechless-data" / "corpus" / "English", corpus_name="dev-clean")


class LabeledExampleTest(TestCase):
    def test(self):
        example = corpus.examples[0]
        mel_power_spectrogram = librosa.feature.melspectrogram(
            y=example.get_raw_audio(), n_fft=example.fourier_window_length, hop_length=example.hop_length,
            sr=example.sample_rate)

        self.assertTrue(np.array_equal(mel_power_spectrogram,
                                       example.spectrogram(type=SpectrogramType.power,
                                                           frequency_scale=SpectrogramFrequencyScale.mel)))

    def test_serialize_positional_label(self):
        a = PositionalLabel(labeled_sections=[("einmal", (0, 0.55555)), ("von", (0.55555, 0.8))])
        s = a.serialize()
        b = PositionalLabel.deserialize(s)

        label, (start, end) = b.labeled_sections[1]
        self.assertEqual("von", label)
        self.assertEqual(0.55555, start)
        self.assertEqual(0.8, end)
