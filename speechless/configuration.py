import logging
from pathlib import Path

# removing this numpy import caused a segfault on startup of tensorflow-gpu 1.0
# see https://github.com/tensorflow/tensorflow/issues/2034
# noinspection PyUnresolvedReferences
import numpy
from collections import OrderedDict
from lazy import lazy
from typing import List, Callable, Optional

from speechless.corpus import LabeledSpectrogramBatchGenerator, Corpus, ComposedCorpus
from speechless.english_corpus import english_corpus, minimal_english_corpus
from speechless.english_corpus import english_frequent_characters
from speechless.german_corpus import german_corpus
from speechless.german_corpus import german_frequent_characters
from speechless.labeled_example import LabeledExampleFromFile
from speechless.net import Wav2Letter, ExpectationsVsPredictionsInGroupedBatches
from speechless.tools import home_directory, timestamp, log, mkdir, write_text, logger


class DataDirectories:
    def __init__(self, data_directory: Path = home_directory() / "speechless-data"):
        self.data_directory = data_directory
        self.corpus_base_directory = data_directory / "corpus"
        self.spectrogram_cache_base_directory = data_directory / "spectrogram-cache"
        self.tensorboard_log_base_directory = data_directory / "logs"
        self.nets_base_directory = data_directory / "nets"
        self.kenlm_base_directory = data_directory / "kenlm"
        self.recording_directory = data_directory / "recordings"
        self.test_results_directory = data_directory / "test-results"


default_data_directories = DataDirectories()


class Configuration:
    def __init__(self,
                 name: str,
                 corpus_from_directory: Callable[[Path], Corpus],
                 allowed_characters: List[chr] = english_frequent_characters,
                 directories: DataDirectories = default_data_directories,
                 mel_frequency_count: int = 128,
                 training_batches_per_epoch: int = 100,
                 batch_size: int = 64):
        self.training_batches_per_epoch = training_batches_per_epoch
        self.mel_frequency_count = mel_frequency_count
        self.name = name
        self.directories = directories
        self.spectrogram_cache_directory = directories.spectrogram_cache_base_directory / name
        self.corpus_directory = directories.corpus_base_directory / name
        self.corpus_from_directory = corpus_from_directory
        self.allowed_characters = allowed_characters
        self.batch_size = batch_size

    @lazy
    def corpus(self) -> Corpus:
        return self.corpus_from_directory(self.corpus_directory)

    @lazy
    def batch_generator(self) -> LabeledSpectrogramBatchGenerator:
        return self.batch_generator_for_corpus(self.corpus)

    def batch_generator_for_corpus(self, corpus: Corpus) -> LabeledSpectrogramBatchGenerator:
        return LabeledSpectrogramBatchGenerator(corpus=corpus,
                                                spectrogram_cache_directory=self.spectrogram_cache_directory,
                                                batch_size=self.batch_size)

    @staticmethod
    def english() -> 'Configuration':
        return Configuration(name="English", corpus_from_directory=english_corpus)

    @staticmethod
    def minimal_english() -> 'Configuration':
        return Configuration(name="English", corpus_from_directory=minimal_english_corpus)

    @staticmethod
    def german(from_cached: bool = True,
               sampled_training_example_count_when_loading_from_cached: Optional[int] = None) -> 'Configuration':
        def load_cached_corpus(corpus_directory: Path) -> Corpus:
            return Corpus.load(corpus_directory / "corpus.csv",
                               sampled_training_example_count=sampled_training_example_count_when_loading_from_cached)

        return Configuration(name="German",
                             allowed_characters=german_frequent_characters,
                             corpus_from_directory=load_cached_corpus if from_cached else german_corpus)

    @staticmethod
    def mixed_german_english():
        return Configuration(
            name="mixed-English-German",
            allowed_characters=german_frequent_characters,
            corpus_from_directory=lambda _: ComposedCorpus(
                [Configuration.english().corpus, Configuration.german().corpus]))

    def train(self, wav2letter, run_name: str) -> None:
        wav2letter.train(self.batch_generator.training_batches(),
                         tensor_board_log_directory=self.directories.tensorboard_log_base_directory / run_name,
                         net_directory=self.directories.nets_base_directory / run_name,
                         preview_labeled_spectrogram_batch=self.batch_generator.preview_batch(),
                         batches_per_epoch=self.training_batches_per_epoch)

    def train_from_beginning(self):
        from speechless.net import Wav2Letter

        wav2letter = Wav2Letter(self.mel_frequency_count, allowed_characters=self.allowed_characters)

        self.train(wav2letter,
                   run_name=timestamp() + "-adam-small-learning-rate-complete-training-{}{}".format(
                       self.name, self.sampled_training_example_count_extension()))

    def summarize_and_save_corpus(self):
        log(self.corpus.summary())
        self.corpus.summarize_to_csv(self.corpus_directory / "summary.csv")
        self.save_corpus()

    def save_corpus(self):
        self.corpus.save(self.corpus_directory / "corpus.csv")

    def fill_cache(self, repair_incorrect: bool = False):
        self.batch_generator.fill_cache(repair_incorrect=repair_incorrect)

    def test_model(self, wav2letter):
        log(wav2letter.test_and_predict_batch(self.batch_generator.preview_batch()))
        log(wav2letter.test_and_predict_batches(self.batch_generator.test_batches()))

    def test_model_grouped_by_loaded_corpus_name(self, wav2letter) -> ExpectationsVsPredictionsInGroupedBatches:
        def corpus_name(example: LabeledExampleFromFile) -> str:
            return example.audio_directory.relative_to(self.corpus_directory).parts[0]

        corpus_by_name = self.corpus.grouped_by(corpus_name)

        log([(name, len(corpus.test_examples)) for name, corpus in corpus_by_name.items()])
        result = wav2letter.test_and_predict_grouped_batches(OrderedDict(
            (corpus_name, self.batch_generator_for_corpus(corpus).test_batches()) for corpus_name, corpus in
            corpus_by_name.items()))
        log(result)

        return result

    def train_transfer_from_best_english_model(self, frozen_layer_count: int,
                                               reinitialize_trainable_loaded_layers: bool = False):
        run_name = timestamp() + "-adam-small-learning-rate-transfer-to-{}-freeze-{}{}{}".format(
            self.name, frozen_layer_count, "-reinitialize" if reinitialize_trainable_loaded_layers else "",
            self.sampled_training_example_count_extension())

        log("Run: " + run_name)

        wav2letter = self.load_best_english_model(
            frozen_layer_count=frozen_layer_count,
            reinitialize_trainable_loaded_layers=reinitialize_trainable_loaded_layers)

        self.train(wav2letter, run_name=run_name)

    def sampled_training_example_count_extension(self):
        return "-{}examples".format(self.corpus.sampled_training_example_count) if \
            self.corpus.sampled_training_example_count is not None else ""

    def load_model(self,
                   load_name: str,
                   load_epoch: int,
                   frozen_layer_count: int = 0,
                   allowed_characters_for_loaded_model: List[chr] = english_frequent_characters,
                   use_kenlm: bool = False,
                   reinitialize_trainable_loaded_layers: bool = False,
                   language_model_name_extension: str = ""):
        from speechless.net import Wav2Letter
        return Wav2Letter(
            allowed_characters=self.allowed_characters,
            input_size_per_time_step=self.mel_frequency_count,
            load_model_from_directory=self.directories.nets_base_directory / load_name,
            load_epoch=load_epoch,
            allowed_characters_for_loaded_model=allowed_characters_for_loaded_model,
            frozen_layer_count=frozen_layer_count,
            kenlm_directory=(
                self.directories.kenlm_base_directory / (
                    self.name.lower() + language_model_name_extension)) if use_kenlm else None,
            reinitialize_trainable_loaded_layers=reinitialize_trainable_loaded_layers)

    def load_best_english_model(self,
                                frozen_layer_count: int = 0,
                                use_ken_lm: bool = False,
                                reinitialize_trainable_loaded_layers: bool = False):
        return self.load_model(
            load_name=Configuration.english_baseline[0], load_epoch=Configuration.english_baseline[1],
            frozen_layer_count=frozen_layer_count,
            use_kenlm=use_ken_lm,
            reinitialize_trainable_loaded_layers=reinitialize_trainable_loaded_layers)

    def test_best_english_model(self, use_kenlm: bool = False):
        self.test_model_grouped_by_loaded_corpus_name(self.load_best_english_model(use_ken_lm=use_kenlm))

    english_baseline = ("20170314-134351-adam-small-learning-rate-complete-95", 1689)

    def test_german_model(self, load_name: str, load_epoch: int, use_ken_lm=False,
                          language_model_name_extension: str = ""):
        self.test_model_grouped_by_loaded_corpus_name(self.load_german_model(
            load_name, load_epoch, use_ken_lm=use_ken_lm, language_model_name_extension=language_model_name_extension))

    def load_german_model(self, load_name: str, load_epoch: int, use_ken_lm=False,
                          language_model_name_extension: str = "") -> Wav2Letter:
        return self.load_model(
            load_name=load_name,
            load_epoch=load_epoch,
            allowed_characters_for_loaded_model=german_frequent_characters,
            use_kenlm=use_ken_lm,
            language_model_name_extension=language_model_name_extension)

    freeze0day4hour7 = ("20170420-001258-adam-small-learning-rate-transfer-to-German-freeze-0", 2066)

    def load_best_german_model(self, use_ken_lm=False,
                               language_model_name_extension: str = "") -> Wav2Letter:
        return self.load_german_model(Configuration.freeze0day4hour7[0], Configuration.freeze0day4hour7[1],
                                      use_ken_lm=use_ken_lm,
                                      language_model_name_extension=language_model_name_extension)

class LoggedRun:
    def __init__(self, action: Callable[[], None], name: str,
                 results_directory: Path = default_data_directories.test_results_directory):
        self.action = action
        self.name = name
        self.results_directory = results_directory
        self.result_file = self.results_directory / self.name

    def __call__(self):
        mkdir(self.results_directory)
        write_text(self.result_file, "")
        handler = logging.FileHandler(str(self.result_file))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            self.action()
        finally:
            logger.removeHandler(handler)
