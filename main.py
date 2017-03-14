import time
from os import makedirs, path
from pathlib import Path

from keras.optimizers import Adam
from numpy import *

from corpus_provider import CorpusProvider
from labeled_example import LabeledExample
from net import Wav2Letter
from recording import Recorder
from spectrogram_batch import LabeledSpectrogramBatchGenerator

base_directory = Path(path.expanduser('~')) / "speechless-data"
base_spectrogram_directory = base_directory / "spectrograms"

# not Path.mkdir() for compatibility with Python 3.4
makedirs(str(base_spectrogram_directory), exist_ok=True)
tensorboard_log_base_directory = base_directory / "logs"
nets_base_directory = base_directory / "nets"
recording_directory = base_directory / "recordings"
corpus_directory = base_directory / "corpus"


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def record():
    corpus = CorpusProvider(corpus_directory, corpus_names=["test-clean"])
    print("Wait in silence to begin recording; wait in silence to terminate")
    # not Path.mkdir() for compatibility with Python 3.4
    makedirs(str(recording_directory), exist_ok=True)
    name = "recording-{}".format(timestamp())
    example = Recorder().record_to_file(recording_directory / "{}.wav".format(name))
    example.save_spectrogram(recording_directory)

    def save_original_spectrogram():
        original = [x for x in corpus.examples if x.id == "6930-75918-0000"][0]
        original.save_spectrogram(recording_directory)

    save_original_spectrogram()


def train_wav2letter() -> None:
    corpus = CorpusProvider(corpus_directory)

    labeled_spectrogram_batch_generator = LabeledSpectrogramBatchGenerator(
        examples=corpus.examples[:int(len(corpus.examples) * .95)],
        spectrogram_cache_directory=base_directory / "spectrogram-cache" / "mel")

    wav2letter = Wav2Letter(input_size_per_time_step=labeled_spectrogram_batch_generator.input_size_per_time_step(),
                            load_model_from_directory=Path(
                                nets_base_directory / "20170314-070930-adam-small-learning-rate-complete-95"),
                            load_epoch=239,
                            optimizer=Adam(1e-4))

    run_name = timestamp() + "-adam-small-learning-rate-complete-95"

    wav2letter.train(labeled_spectrogram_batch_generator.training_batches(),
                     tensor_board_log_directory=tensorboard_log_base_directory / run_name,
                     net_directory=nets_base_directory / run_name,
                     test_labeled_spectrogram_batch=labeled_spectrogram_batch_generator.test_batch(),
                     samples_per_epoch=labeled_spectrogram_batch_generator.batch_size * 100)


def predict_recording() -> None:
    wav2letter = Wav2Letter(
        input_size_per_time_step=128,
        load_model_from_directory=Path(nets_base_directory / "20170314-070930-adam-small-learning-rate-complete-95"),
        load_epoch=239)

    def print_prediction(name: str) -> None:
        sample = LabeledExample.from_file(recording_directory / name)
        print(wav2letter.predict_single(sample.z_normalized_transposed_spectrogram()))

    print_prediction("6930-75918-0000.flac")
    print_prediction("recording-20170310-135534.wav")
    print_prediction("recording-20170310-135144.wav")


train_wav2letter()