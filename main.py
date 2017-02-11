import time
from os import makedirs, path
from pathlib import Path

from keras.optimizers import Adagrad
from numpy import *

from corpus_provider import CorpusProvider
from net import Wav2Letter
from spectrogram_batch import LabeledSpectrogramBatchGenerator

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
    labels = [
        "thank you dorcas dear",
        "yes rachel i do love you",
        "dorcas in her strange way was moved",
        "you have been so ill my poor rachel",
        "thank you rachel my cousin rachel my only friend",
        "well she was better though she had had a bad night",
        "i like you still rachel i'm sure i'll always like you",
        "ill and troubled dear troubled in mind and miserably nervous",
        "you resemble me rachel you are fearless and inflexible and generous",
        "this transient spring and lighting up are beautiful a glamour beguiling our senses",
        "and she threw her arms round her cousin's neck and brave rachel at last burst into tears",
        "i have very few to love me now and i thought you might love me as i have begun to love you",
        "it is an antipathy an antipathy i cannot get over dear dorcas you may think it a madness but don't blame me",
        "yes something everything said rachel hurriedly looking frowningly at a flower which she was twirling in her fingers",
        "women can hide their pain better than we men and bear it better too except when shame drops fire into the dreadful chalice",
        "and the wan oracle having spoken she sate down in the same sort of abstraction again beside dorcas and she looked full in her cousin's eyes",
        "but poor rachel lake had more than that stoical hypocrisy which enables the tortured spirits of her sex to lift a pale face through the flames and smile",
        "so there came a step and a little rustling of feminine draperies the small door opened and rachel entered with her hand extended and a pale smile of welcome",
        "chelford had a note from mister wylder this morning another note his coming delayed and something of his having to see some person who is abroad continued dorcas after a little pause",
        "there was something of sweetness and fondness in her tones and manner which was new to rachel and comforting and she returned the greeting as kindly and felt more like her former self"]

    return sorted([example for example in corpus.examples if example.label.lower() in labels],
                  key=lambda x: len(x.label))


labeled_spectrogram_batch_generator = LabeledSpectrogramBatchGenerator(
    examples=corpus.examples[:int(len(corpus.examples) * .95)],
    spectrogram_cache_directory=base_directory / "spectrogram-cache" / "mel")


def train_wav2letter() -> None:
    wav2letter = Wav2Letter(input_size_per_time_step=labeled_spectrogram_batch_generator.input_size_per_time_step(),
                            optimizer=Adagrad(lr=1e-3))
    name = timestamp() + "-adagrad-complete-95"

    wav2letter.train(labeled_spectrogram_batch_generator.training_batches(),
                     tensor_board_log_directory=tensorboard_log_base_directory / name,
                     net_directory=nets_base_directory / name,
                     test_labeled_spectrogram_batch=labeled_spectrogram_batch_generator.test_batch(),
                     samples_per_epoch=labeled_spectrogram_batch_generator.batch_size * 100)


train_wav2letter()
