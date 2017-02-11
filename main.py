import time
from os import makedirs, path
from pathlib import Path
from typing import List

from keras.optimizers import Adagrad
from numpy import *

from corpus_provider import CorpusProvider
from labeled_example import LabeledExample
from net import Wav2Letter

base_directory = Path(path.expanduser('~')) / "speechless-data"
base_spectrogram_directory = base_directory / "spectrograms"

# not Path.mkdir() for compatibility with Python 3.4
makedirs(str(base_spectrogram_directory), exist_ok=True)
tensorboard_log_base_directory = base_directory / "logs"
nets_base_directory = base_directory / "nets"


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


corpus = CorpusProvider(base_directory / "corpus")


def first_20_examples_sorted_by_length():
    return sorted(corpus.examples[:20], key=lambda x: len(x.label))


def train_wav2letter(examples: List[LabeledExample]) -> None:
    spectrograms = [example.z_normalized_transposed_spectrogram() for example in examples]

    wav2letter = Wav2Letter(input_size_per_time_step=spectrograms[0].shape[1], optimizer=Adagrad(lr=1e-3))
    name = timestamp() + "-adagrad"

    wav2letter.train(spectrograms=spectrograms, labels=[example.label for example in examples],
                     tensor_board_log_directory=tensorboard_log_base_directory / name,
                     net_directory=nets_base_directory / name)


train_wav2letter(first_20_examples_sorted_by_length())
