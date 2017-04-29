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
from speechless.english_corpus import english_corpus
from speechless.german_corpus import german_corpus
from speechless.grapheme_enconding import english_frequent_characters, german_frequent_characters
from speechless.labeled_example import LabeledExample
from speechless.net import Wav2Letter
from speechless.tools import home_directory, timestamp, log, mkdir, write_text, logger

base_directory = home_directory() / "speechless-data"
tensorboard_log_base_directory = base_directory / "logs"
nets_base_directory = base_directory / "nets"
recording_directory = base_directory / "recordings"
corpus_base_directory = base_directory / "corpus"
spectrogram_cache_base_directory = base_directory / "spectrogram-cache"
kenlm_base_directory = base_directory / "kenlm"


class Configuration:
    def __init__(self,
                 name: str,
                 allowed_characters: List[chr],
                 corpus_from_directory: Callable[[Path], Corpus],
                 corpus_directory: Optional[Path] = None,
                 spectrogram_cache_directory: Optional[Path] = None,
                 mel_frequency_count: int = 128,
                 training_batches_per_epoch: int = 100,
                 batch_size: int = 64):
        self.training_batches_per_epoch = training_batches_per_epoch
        self.mel_frequency_count = mel_frequency_count
        self.name = name
        self.spectrogram_cache_directory = spectrogram_cache_directory if spectrogram_cache_directory else \
            spectrogram_cache_base_directory / name
        self.corpus_directory = corpus_directory if corpus_directory else corpus_base_directory / name
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
        return Configuration(name="English",
                             allowed_characters=english_frequent_characters,
                             corpus_from_directory=english_corpus)

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
                         tensor_board_log_directory=tensorboard_log_base_directory / run_name,
                         net_directory=nets_base_directory / run_name,
                         preview_labeled_spectrogram_batch=self.batch_generator.preview_batch(),
                         batches_per_epoch=self.training_batches_per_epoch)

    def train_from_beginning(self):
        from speechless.net import Wav2Letter

        wav2letter = Wav2Letter(self.mel_frequency_count, allowed_characters=self.allowed_characters)

        self.train(wav2letter, run_name=timestamp() + "-adam-small-learning-rate-complete-training-{}{}".format(
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

    def test_model_grouped_by_loaded_corpus_name(self, wav2letter):
        def corpus_name(example: LabeledExample) -> str:
            return example.audio_directory.relative_to(self.corpus_directory).parts[0]

        corpus_by_name = self.corpus.grouped_by(corpus_name)

        log([(name, len(corpus.test_examples)) for name, corpus in corpus_by_name.items()])
        log(wav2letter.test_and_predict_grouped_batches(OrderedDict(
            (corpus_name, self.batch_generator_for_corpus(corpus).test_batches()) for corpus_name, corpus in
            corpus_by_name.items())))

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
            load_model_from_directory=nets_base_directory / load_name,
            load_epoch=load_epoch,
            allowed_characters_for_loaded_model=allowed_characters_for_loaded_model,
            frozen_layer_count=frozen_layer_count,
            kenlm_directory=(
                kenlm_base_directory / (self.name.lower() + language_model_name_extension)) if use_kenlm else None,
            reinitialize_trainable_loaded_layers=reinitialize_trainable_loaded_layers)

    def load_best_english_model(self,
                                frozen_layer_count: int = 0,
                                use_ken_lm: bool = False,
                                reinitialize_trainable_loaded_layers: bool = False):
        return self.load_model(
            load_name="20170314-134351-adam-small-learning-rate-complete-95", load_epoch=1689,
            frozen_layer_count=frozen_layer_count,
            use_kenlm=use_ken_lm,
            reinitialize_trainable_loaded_layers=reinitialize_trainable_loaded_layers)

    def test_best_english_model(self, use_kenlm: bool = False):
        self.test_model_grouped_by_loaded_corpus_name(self.load_best_english_model(use_ken_lm=use_kenlm))

    def load_best_english_model_trained_in_one_run(self):
        return self.load_model(
            load_name="20170316-180957-adam-small-learning-rate-complete-95", load_epoch=1192)

    def test_best_english_model_trained_in_one_run(self):
        self.test_model(self.load_best_english_model_trained_in_one_run())

    freeze0day4hour7 = ("20170420-001258-adam-small-learning-rate-transfer-to-German-freeze-0", 2066)
    german_from_beginning = ("20170415-001150-adam-small-learning-rate-complete-training-German", 443)

    english_baseline = ("20170314-134351-adam-small-learning-rate-complete-95", 1689)
    english_correct_test_split = ("20170414-113509-adam-small-learning-rate-complete-training", 733)

    freeze0 = ("20170420-001258-adam-small-learning-rate-transfer-to-German-freeze-0", 1704)
    freeze6 = ("20170419-212024-adam-small-learning-rate-transfer-to-German-freeze-6", 1708)
    freeze8 = ("20170418-120145-adam-small-learning-rate-transfer-to-German-freeze-8", 1759)
    freeze9 = ("20170419-235043-adam-small-learning-rate-transfer-to-German-freeze-9", 1789)
    freeze10 = ("20170415-092748-adam-small-learning-rate-transfer-to-German-freeze-10", 1778)


    freeze8reinitialize = ("20170418-140152-adam-small-learning-rate-transfer-to-German-freeze-8-reinitialize", 1755)
    freeze8small = ("20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 1809)
    freeze8small_15hours = ("20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 1727)
    freeze8small_20hours = ("20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 1767)
    freeze8small_40hours = ("20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 1939)
    freeze8small_50hours = ("20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 2021)
    freeze8tiny = ("20170424-231220-adam-small-learning-rate-transfer-to-German-freeze-8-10000examples", 1844)
    freeze8tiny_1742 = ("20170424-231220-adam-small-learning-rate-transfer-to-German-freeze-8-10000examples", 1742)
    freeze8tiny_1716 = ("20170424-231220-adam-small-learning-rate-transfer-to-German-freeze-8-10000examples", 1716)

    german_small_from_beginning_day2hour15 = \
        ("20170424-232706-adam-small-learning-rate-complete-training-German-50000examples", 237)
    freeze8small_day2hour15 = \
        ("20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 2121)

    german_model_names_with_epochs = [freeze0day4hour7, german_from_beginning, freeze0, freeze6, freeze8, freeze9,
                                      freeze10, freeze8reinitialize,
                                      freeze8small, freeze8small_15hours, freeze8small_20hours,
                                      freeze8small_day2hour15, freeze8small_40hours, freeze8small_50hours,
                                      freeze8tiny, freeze8tiny_1742, freeze8tiny_1716,
                                      german_small_from_beginning_day2hour15]

    def test_german_model(self, load_name: str, load_epoch: int, use_ken_lm=True,
                          language_model_name_extension: str = ""):
        self.test_model_grouped_by_loaded_corpus_name(self.load_german_model(
            load_name, load_epoch, use_ken_lm=use_ken_lm, language_model_name_extension=language_model_name_extension))

    def load_german_model(self, load_name: str, load_epoch: int, use_ken_lm=True,
                          language_model_name_extension: str = "") -> Wav2Letter:
        return self.load_model(
            load_name=load_name,
            load_epoch=load_epoch,
            allowed_characters_for_loaded_model=german_frequent_characters,
            use_kenlm=use_ken_lm,
            language_model_name_extension=language_model_name_extension)


class LoggedRun:
    def __init__(self, action: Callable[[], None], name: str,
                 results_directory: Path = base_directory / "test-results"):
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
