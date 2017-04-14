from pathlib import Path

# removing this numpy import caused a segfault on startup of tensorflow-gpu 1.0
# see https://github.com/tensorflow/tensorflow/issues/2034
# noinspection PyUnresolvedReferences
import numpy
from lazy import lazy
from typing import List, Callable

from corpus import LabeledSpectrogramBatchGenerator, Corpus
from english_corpus import english_corpus
from german_corpus import german_corpus, sc10
from grapheme_enconding import english_frequent_characters, german_frequent_characters
from labeled_example import LabeledExample, LabeledSpectrogram
from recording import Recorder
from tools import mkdir, home_directory, timestamp

base_directory = home_directory() / "speechless-data"
tensorboard_log_base_directory = base_directory / "logs"
nets_base_directory = base_directory / "nets"
recording_directory = base_directory / "recordings"

corpus_base_directory = base_directory / "corpus"
english_corpus_directory = corpus_base_directory / "English"
german_corpus_directory = corpus_base_directory / "German"

spectrogram_cache_base_directory = base_directory / "spectrogram-cache"
english_spectrogram_cache_directory = spectrogram_cache_base_directory / "English"
german_spectrogram_cache_directory = spectrogram_cache_base_directory / "German"


def load_cached_corpus(corpus_directory: Path) -> Corpus:
    return Corpus.load(corpus_directory / "corpus.csv")


class Configuration:
    def __init__(self,
                 name: str,
                 allowed_characters: english_frequent_characters,
                 corpus_from_directory: Callable[[Path], Corpus],
                 corpus_directory: Path,
                 spectrogram_cache_directory: Path):
        self.name = name
        self.spectrogram_cache_directory = spectrogram_cache_directory
        self.corpus_directory = corpus_directory
        self.corpus_from_directory = corpus_from_directory
        self.allowed_characters = allowed_characters

    @lazy
    def corpus(self) -> Corpus:
        return self.corpus_from_directory(self.corpus_directory)

    @lazy
    def batch_generator(self) -> LabeledSpectrogramBatchGenerator:
        return LabeledSpectrogramBatchGenerator(corpus=self.corpus,
                                                spectrogram_cache_directory=self.spectrogram_cache_directory)

    @staticmethod
    def english() -> 'Configuration':
        return Configuration(name="English",
                             allowed_characters=english_frequent_characters,
                             corpus_from_directory=english_corpus,
                             corpus_directory=english_corpus_directory,
                             spectrogram_cache_directory=english_spectrogram_cache_directory)

    @staticmethod
    def german(from_cached: bool = True) -> 'Configuration':
        return Configuration(name="German",
                             allowed_characters=german_frequent_characters,
                             corpus_from_directory=load_cached_corpus if from_cached else german_corpus,
                             corpus_directory=german_corpus_directory,
                             spectrogram_cache_directory=german_spectrogram_cache_directory)

    def train(self):
        from net_with_corpus import Wav2LetterWithCorpus
        from net import Wav2Letter

        mel_frequency_count = 128
        batches_per_epoch = 100
        run_name = timestamp() + "-adam-small-learning-rate-complete-training-{}".format(self.name)

        wav2letter = Wav2Letter(mel_frequency_count, allowed_characters=self.allowed_characters)

        wav2letter_with_corpus = Wav2LetterWithCorpus(wav2letter, self.corpus,
                                                      spectrogram_cache_directory=self.spectrogram_cache_directory)

        wav2letter_with_corpus.train(tensor_board_log_directory=tensorboard_log_base_directory / run_name,
                                     net_directory=nets_base_directory / run_name,
                                     batches_per_epoch=batches_per_epoch)

    def summarize_and_save_corpus(self):
        corpus = self.corpus
        print(corpus.summary())
        corpus.summarize_to_csv(self.corpus_directory / "summary.csv")
        corpus.save(self.corpus_directory / "corpus.csv")

    def fill_up_cache(self):
        self.batch_generator.fill_cache()

    def test_best_model(self):
        wav2letter = load_best_wav2letter_model(allowed_characters=self.allowed_characters)

        print(wav2letter.expectations_vs_predictions(self.batch_generator.preview_batch()))
        print("Average loss: {}".format(wav2letter.loss(self.batch_generator.test_batches())))


def train_transfer_english_to_german():
    from net_with_corpus import Wav2LetterWithCorpus

    batch_size = 64
    mel_frequency_count = 128
    batches_per_epoch = 100
    layer_count = 11
    trainable_layer_count = 1
    frozen_layer_count = layer_count - trainable_layer_count
    run_name = timestamp() + "-adam-small-learning-rate-transfer-sc10-freeze-{}".format(frozen_layer_count)

    wav2letter = load_best_wav2letter_model(mel_frequency_count, allowed_characters=german_frequent_characters,
                                            frozen_layer_count=frozen_layer_count)

    corpus = sc10(german_corpus_directory, training_test_split=lambda examples: (examples[:batch_size], []))

    wav2letter_with_corpus = Wav2LetterWithCorpus(wav2letter, corpus, batch_size=batch_size,
                                                  spectrogram_cache_directory=german_spectrogram_cache_directory)

    wav2letter_with_corpus.train(tensor_board_log_directory=tensorboard_log_base_directory / run_name,
                                 net_directory=nets_base_directory / run_name,
                                 batches_per_epoch=batches_per_epoch)


def load_best_wav2letter_model(mel_frequency_count: int = 128,
                               frozen_layer_count=0,
                               allowed_characters: List[chr] = english_frequent_characters):
    from net import Wav2Letter

    return Wav2Letter(
        allowed_characters=allowed_characters,
        input_size_per_time_step=mel_frequency_count,
        load_model_from_directory=nets_base_directory / "20170314-134351-adam-small-learning-rate-complete-95",
        load_epoch=1689,
        allowed_characters_for_loaded_model=english_frequent_characters,
        frozen_layer_count=frozen_layer_count)


def record_plot_and_save() -> LabeledExample:
    from labeled_example_plotter import LabeledExamplePlotter

    print("Wait in silence to begin recording; wait in silence to terminate")
    mkdir(recording_directory)
    name = "recording-{}".format(timestamp())
    example = Recorder().record_to_file(recording_directory / "{}.wav".format(name))
    LabeledExamplePlotter(example).save_spectrogram(recording_directory)

    return example


def predict_recording() -> None:
    wav2letter = load_best_wav2letter_model()

    def predict(labeled_spectrogram: LabeledSpectrogram) -> str:
        return wav2letter.predict_single(labeled_spectrogram.z_normalized_transposed_spectrogram())

    def print_prediction(labeled_spectrogram: LabeledSpectrogram, description: str = None) -> None:
        print((description if description else labeled_spectrogram.id) + ": " + '"{}"'.format(
            predict(labeled_spectrogram)))

    def record_and_print_prediction(description: str = None) -> None:
        print_prediction(record_plot_and_save(), description=description)

    def print_prediction_from_file(file_name: str, description: str = None) -> None:
        print_prediction(LabeledExample(recording_directory / file_name), description=description)

    def print_example_predictions() -> None:
        print("Predictions: ")
        print_prediction_from_file("6930-75918-0000.flac",
                                   description='Sample labeled "concord returned to its place amidst the tents"')
        print_prediction_from_file("recording-20170310-135534.wav", description="Recorded playback of the same sample")
        print_prediction_from_file("recording-20170310-135144.wav",
                                   description="Recording of me saying the same sentence")
        print_prediction_from_file("recording-20170314-224329.wav",
                                   description='Recording of me saying "I just wrote a speech recognizer"')
        print_prediction_from_file("bad-quality-louis.wav",
                                   description="Louis' rerecording of worse quality")

    print_example_predictions()


def summarize_and_save_german_corpus():
    Configuration.german(from_cached=False).summarize_and_save_corpus()


def summarize_and_save_english_corpus():
    Configuration.english().summarize_and_save_corpus()


# summarize_and_save_german_corpus()

# Configuration.german().fill_up_cache()

Configuration.german().test_best_model()

# summarize_and_save_english_corpus()

# Configuration.german().train()

# LabeledExampleTest().test()

# Configuration.german().train()
