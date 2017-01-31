import time
from os import makedirs, path
from pathlib import Path
from typing import List

from numpy import *

from corpus_provider import CorpusProvider
from labeled_example import LabeledExample, SpectrogramFrequencyScale, SpectrogramType
from net import Wav2Letter

base_directory = Path(path.expanduser('~')) / "speechless-data"
base_spectrogram_directory = base_directory / "spectrograms"

# not Path.mkdir() for compatibility with Python 3.4
makedirs(str(base_spectrogram_directory), exist_ok=True)
tensorboard_log_base_directory = base_directory / "logs"


def tensorboard_log_directory_timestamped() -> Path:
    return Path(tensorboard_log_base_directory, time.strftime("%Y%m%d-%H%M%S"))


corpus = CorpusProvider(base_directory / "corpus")


def first_20_examples_sorted_by_length():
    return sorted(corpus.examples[:20], key=lambda x: len(x.label))


def z_normalize(array: ndarray) -> ndarray:
    return (array - mean(array)) / std(array)


def normalized_transposed_spectrograms(examples: List[LabeledExample]) -> List[ndarray]:
    """

    :param examples:
    :return: Array with shape (time, frequencies), z-normalized
    """
    return [z_normalize(example.spectrogram(frequency_scale=SpectrogramFrequencyScale.mel).T) for example in examples]


def labels(examples: List[LabeledExample]):
    return [example.label for example in examples]


def train_wav2letter(examples: List[LabeledExample]):
    spectrograms = normalized_transposed_spectrograms(examples)

    # TODO check again whether output = softmax is necessary, as tf.ctc_loss states:
    # "This class performs the softmax operation for you"
    # (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard0/tf.nn.ctc_loss.md)
    # vs. softmax activation unit here:
    # https://github.com/fchollet/keras/blob/883f74ca410e822fba266c4c344a09e364693951/examples/image_ocr.py#L456

    # TODO also check: for some reason, the keras implementation takes the logarithm of the predictions
    # (see keras.backend.ctc_batch_cost)

    wav2letter = Wav2Letter(input_size_per_time_step=spectrograms[0].shape[1], output_activation="softmax")
    wav2letter.train(spectrograms=spectrograms, labels=labels(examples),
                     tensor_board_log_directory=tensorboard_log_directory_timestamped())


def save_spectrograms_of_all_types(example: LabeledExample):
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

train_wav2letter(first_20_examples_sorted_by_length())
