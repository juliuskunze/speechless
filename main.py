import time
from pathlib import Path
from typing import List

from keras.engine import Model
from numpy import *

from corpus_provider import CorpusProvider
from labeled_example import LabeledExample, SpectrogramFrequencyScale, SpectrogramType
from net import wav2letter_net, wav2letter_trained_on_batch, ctc_grapheme_set_size

# not Path.home() for compatibility with Python 3.4
base_directory = Path(os.path.expanduser('~'), "speechless-data")
base_spectrogram_directory = Path(base_directory, "spectrograms")

# not Path.mkdir() for compatibility with Python 3.4
os.makedirs(str(base_spectrogram_directory), exist_ok=True)
tensorboard_log_base_directory = Path(base_directory, 'logs')


def tensorboard_log_directory_timestamped() -> Path:
    return Path(tensorboard_log_base_directory, time.strftime("%Y%m%d-%H%M%S"))


corpus = CorpusProvider(Path(base_directory, "corpus"))


def first_20_examples_sorted_by_length():
    return sorted(corpus.examples[:20], key=lambda x: len(x.label))


def znormalize(array: ndarray) -> ndarray:
    return (array - mean(array)) / std(array)


def spectrograms(examples: List[LabeledExample]) -> List[ndarray]:
    """

    :param examples:
    :return: Array with dimensions (time, frequencies), z-normalized
    """
    return [znormalize(example.spectrogram(frequency_scale=SpectrogramFrequencyScale.mel).T) for example in examples]


def labels(examples: List[LabeledExample]):
    return [example.label for example in examples]


examples = first_20_examples_sorted_by_length()


def trained_wav2letter() -> Model:
    s = spectrograms(examples)

    input_size_per_time_step = s[0].shape[1]

    # TODO check again whether output = softmax is necessary, as ctcloss states:
    # "This class performs the softmax operation for you"
    # (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard0/tf.nn.ctc_loss.md)
    net = wav2letter_net(input_size_per_time_step=input_size_per_time_step,
                         output_grapheme_set_size=ctc_grapheme_set_size, output_activation="softmax")
    print(net.summary())

    return wav2letter_trained_on_batch(net, spectrograms=s,
                                       labels=labels(examples),
                                       tensor_board_log_directory=tensorboard_log_directory_timestamped())


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

model = trained_wav2letter()
