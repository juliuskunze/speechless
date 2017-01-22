from pathlib import Path
from typing import List

from keras.engine import Model

from corpus_provider import CorpusProvider
from labeled_example import LabeledExample, SpectrogramFrequencyScale, SpectrogramType
from net import wav2letter_net, wav2letter_trained_on_batch

base_directory = Path(Path.home(), "speechless-data")
base_spectrogram_directory = Path(base_directory, "spectrograms")
base_spectrogram_directory.mkdir(exist_ok=True)
corpus = CorpusProvider(Path(base_directory, "corpus"))
shortest_10_out_of_20 = sorted(corpus.examples[:20], key=lambda x: len(x.label))[:10]


def spectrograms(examples: List[LabeledExample]):
    return [example.spectrogram(frequency_scale=SpectrogramFrequencyScale.mel).T for example in examples]


def labels(examples: List[LabeledExample]):
    return [example.label for example in examples]


def trained_wav2letter() -> Model:
    net = wav2letter_net(input_size_per_timestep=5000, output_charset_size=40, output_activation="softmax")
    print(net.summary())

    return wav2letter_trained_on_batch(net, spectrograms=spectrograms(shortest_10_out_of_20),
                                       labels=labels(shortest_10_out_of_20))


def save_spectrograms(example: LabeledExample):
    for type in SpectrogramType:
        for frequency_scale in SpectrogramFrequencyScale:
            example.save_spectrogram(target_directory=base_spectrogram_directory, type=type,
                                     frequency_scale=frequency_scale)


def save_first_ten_spectrograms():
    for i in range(10):
        corpus.examples[i].save_spectrogram(target_directory=base_spectrogram_directory)

        # corpus.examples[0].save_spectrogram(base_spectrogram_directory)
        # corpus.examples[0].plot_raw_sound()
        # save_spectrograms(corpus.examples[0])
        # save_first_ten_spectrograms()
