from pathlib import Path

# removing this numpy import caused a segfault on startup of tensorflow-gpu 1.0
# see https://github.com/tensorflow/tensorflow/issues/2034
# noinspection PyUnresolvedReferences
import numpy
from lazy import lazy
from typing import List, Callable

from speechless.corpus import LabeledSpectrogramBatchGenerator, Corpus
from speechless.english_corpus import english_corpus
from speechless.german_corpus import german_corpus
from speechless.grapheme_enconding import english_frequent_characters, german_frequent_characters
from speechless.tools import home_directory, timestamp

base_directory = home_directory() / "speechless-data"
tensorboard_log_base_directory = base_directory / "logs"
nets_base_directory = base_directory / "nets"
recording_directory = base_directory / "recordings"
corpus_base_directory = base_directory / "corpus"
spectrogram_cache_base_directory = base_directory / "spectrogram-cache"
kenlm_base_directory = base_directory / "kenlm"


def load_cached_corpus(corpus_directory: Path) -> Corpus:
    return Corpus.load(corpus_directory / "corpus.csv")


class Configuration:
    def __init__(self,
                 name: str,
                 allowed_characters: List[chr],
                 corpus_from_directory: Callable[[Path], Corpus],
                 corpus_directory: Path = None,
                 spectrogram_cache_directory: Path = None,
                 mel_frequency_count: int = 128,
                 training_batches_per_epoch: int = 100):
        self.training_batches_per_epoch = training_batches_per_epoch
        self.mel_frequency_count = mel_frequency_count
        self.name = name
        self.spectrogram_cache_directory = spectrogram_cache_directory if spectrogram_cache_directory else \
            spectrogram_cache_base_directory / name
        self.corpus_directory = corpus_directory if corpus_directory else corpus_base_directory / name
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
                             corpus_from_directory=english_corpus)

    @staticmethod
    def german(from_cached: bool = True) -> 'Configuration':
        return Configuration(name="German",
                             allowed_characters=german_frequent_characters,
                             corpus_from_directory=load_cached_corpus if from_cached else german_corpus)

    def train(self):
        from speechless.net_with_corpus import Wav2LetterWithCorpus
        from speechless.net import Wav2Letter

        run_name = timestamp() + "-adam-small-learning-rate-complete-training-{}".format(self.name)

        wav2letter = Wav2Letter(self.mel_frequency_count, allowed_characters=self.allowed_characters)

        wav2letter_with_corpus = Wav2LetterWithCorpus(wav2letter, self.corpus,
                                                      spectrogram_cache_directory=self.spectrogram_cache_directory)

        wav2letter_with_corpus.train(tensor_board_log_directory=tensorboard_log_base_directory / run_name,
                                     net_directory=nets_base_directory / run_name,
                                     batches_per_epoch=self.training_batches_per_epoch)

    def summarize_and_save_corpus(self):
        print(self.corpus.summary())
        self.corpus.summarize_to_csv(self.corpus_directory / "summary.csv")
        self.save_corpus()

    def save_corpus(self):
        self.corpus.save(self.corpus_directory / "corpus.csv")

    def fill_cache(self, repair_incorrect: bool = False):
        self.batch_generator.fill_cache(repair_incorrect=repair_incorrect)

    def test_model(self, wav2letter):
        print(wav2letter.test_and_predict_batch(self.batch_generator.preview_batch()))
        print(wav2letter.test_and_predict_batches(self.batch_generator.test_batches()))

    def train_transfer_from_best_english_model(self, trainable_layer_count: int = 1,
                                               reinitialize_trainable_loaded_layers: bool = False):
        from speechless.net_with_corpus import Wav2LetterWithCorpus

        layer_count = 11
        frozen_layer_count = layer_count - trainable_layer_count
        run_name = timestamp() + "-adam-small-learning-rate-transfer-to-{}-freeze-{}{}".format(
            self.name, frozen_layer_count, "-reinitialize" if reinitialize_trainable_loaded_layers else "")

        wav2letter = self.load_best_english_model(
            frozen_layer_count=frozen_layer_count,
            reinitialize_trainable_loaded_layers=reinitialize_trainable_loaded_layers)

        wav2letter_with_corpus = Wav2LetterWithCorpus(wav2letter, self.corpus,
                                                      spectrogram_cache_directory=self.spectrogram_cache_directory)

        wav2letter_with_corpus.train(tensor_board_log_directory=tensorboard_log_base_directory / run_name,
                                     net_directory=nets_base_directory / run_name,
                                     batches_per_epoch=self.training_batches_per_epoch)

    def load_model(self,
                   load_name: str,
                   load_epoch: int,
                   frozen_layer_count: int = 0,
                   allowed_characters_for_loaded_model: List[chr] = english_frequent_characters,
                   use_ken_lm: bool = False,
                   reinitialize_trainable_loaded_layers: bool = False):
        from speechless.net import Wav2Letter

        return Wav2Letter(
            allowed_characters=self.allowed_characters,
            input_size_per_time_step=self.mel_frequency_count,
            load_model_from_directory=nets_base_directory / load_name,
            load_epoch=load_epoch,
            allowed_characters_for_loaded_model=allowed_characters_for_loaded_model,
            frozen_layer_count=frozen_layer_count,
            kenlm_directory=(kenlm_base_directory / self.name.lower()) if use_ken_lm else None,
            reinitialize_trainable_loaded_layers=reinitialize_trainable_loaded_layers)

    def load_best_english_model(self,
                                frozen_layer_count: int = 0,
                                use_ken_lm: bool = False,
                                reinitialize_trainable_loaded_layers: bool = False):
        return self.load_model(
            load_name="20170314-134351-adam-small-learning-rate-complete-95", load_epoch=1689,
            frozen_layer_count=frozen_layer_count,
            use_ken_lm=use_ken_lm,
            reinitialize_trainable_loaded_layers=reinitialize_trainable_loaded_layers)

    def test_best_english_model(self, use_ken_lm: bool = False):
        self.test_model(self.load_best_english_model(use_ken_lm=use_ken_lm))

    def load_best_english_model_trained_in_one_run(self):
        return self.load_model(
            load_name="20170316-180957-adam-small-learning-rate-complete-95", load_epoch=1192)

    def test_best_english_model_trained_in_one_run(self):
        self.test_model(self.load_best_english_model_trained_in_one_run())
