import sys
from pathlib import Path
from socket import gethostname

from typing import List, Tuple

from speechless import configuration, german_corpus
from speechless.configuration import Configuration, LoggedRun
from speechless.german_corpus import german_frequent_characters
from speechless.net import ExpectationsVsPredictionsInGroupedBatches
from speechless.tools import log


def restrict_gpu_memory(per_process_gpu_memory_fraction: float = 0.9):
    import os
    import tensorflow as tf
    import keras
    thread_count = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            allow_soft_placement=True,
                            intra_op_parallelism_threads=thread_count) \
        if thread_count else tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


if __name__ == '__main__':
    class OldRuns:
        freeze0day4hour7 = ("20170420-001258-adam-small-learning-rate-transfer-to-German-freeze-0", 2066)
        german_from_beginning = ("20170415-001150-adam-small-learning-rate-complete-training-German", 443)

        english_baseline = ("20170314-134351-adam-small-learning-rate-complete-95", 1689)
        english_correct_test_split = ("20170414-113509-adam-small-learning-rate-complete-training", 733)

        english_baseline_in_one_run = ("20170316-180957-adam-small-learning-rate-complete-95", 1192)

        freeze0 = ("20170420-001258-adam-small-learning-rate-transfer-to-German-freeze-0", 1704)
        freeze6 = ("20170419-212024-adam-small-learning-rate-transfer-to-German-freeze-6", 1708)
        freeze8 = ("20170418-120145-adam-small-learning-rate-transfer-to-German-freeze-8", 1759)
        freeze9 = ("20170419-235043-adam-small-learning-rate-transfer-to-German-freeze-9", 1789)
        freeze10 = ("20170415-092748-adam-small-learning-rate-transfer-to-German-freeze-10", 1778)

        freeze8reinitialize = (
            "20170418-140152-adam-small-learning-rate-transfer-to-German-freeze-8-reinitialize", 1755)
        freeze8small = ("20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 1809)
        freeze8small_15hours = (
            "20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 1727)
        freeze8small_20hours = (
            "20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 1767)
        freeze8small_40hours = (
            "20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 1939)
        freeze8small_50hours = (
            "20170420-174046-adam-small-learning-rate-transfer-to-German-freeze-8-50000examples", 2021)
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


    class EluRuns:
        english_elu_500 = ("20170509-140404-adam-small-learning-rate-complete-training-English-elu", 500)
        english_elu_750 = ("20170509-140404-adam-small-learning-rate-complete-training-English-elu", 750)
        english_elu_1000 = ("20170509-140404-adam-small-learning-rate-complete-training-English-elu", 1000)
        english_elu_1250 = ("20170509-140404-adam-small-learning-rate-complete-training-English-elu", 1250)
        english_elu_1500 = ("20170509-140404-adam-small-learning-rate-complete-training-English-elu", 1500)
        english_elu_2000 = ("20170509-140404-adam-small-learning-rate-complete-training-English-elu", 2000)
        english_elu_3000 = ("20170509-140404-adam-small-learning-rate-complete-training-English-elu", 3000)


    if gethostname() == "ketos":
        ketos_spectrogram_cache_base_directory = configuration.default_data_directories.data_directory / "ketos-spectrogram-cache"
        ketos_kenlm_base_directory = configuration.default_data_directories.data_directory / "ketos-kenlm"

        log("Running on ketos, using spectrogram cache base directory {} and kenlm base directory {}".format(
            ketos_spectrogram_cache_base_directory, ketos_kenlm_base_directory))
        configuration.default_data_directories.spectrogram_cache_base_directory = ketos_spectrogram_cache_base_directory
        configuration.default_data_directories.kenlm_base_directory = ketos_kenlm_base_directory
    else:
        restrict_gpu_memory()


    # Configuration.german().train_transfer_from_best_english_model(frozen_layer_count=6)

    # Configuration.german().train_from_beginning()

    # Configuration.german(from_cached=False).summarize_and_save_corpus()

    # Configuration.german().fill_cache(repair_incorrect=True)

    # Configuration.german().test_best_model()

    # Configuration.english().summarize_and_save_corpus()

    # Configuration.german(sampled_training_example_count_when_loading_from_cached=50000).train_from_beginning()

    # net = Configuration.english().load_best_english_model().predictive_net

    # Configuration.german(sampled_training_example_count_when_loading_from_cached=10000). \
    #    train_transfer_from_best_english_model(frozen_layer_count=8)

    # Configuration.english().save_corpus()

    # Configuration.mixed_german_english().train_from_beginning()

    # Configuration.english().train_from_beginning()

    def summarize_and_save_small():
        Configuration(name="German",
                      allowed_characters=german_frequent_characters,
                      corpus_from_directory=german_corpus.sc10).summarize_and_save_corpus()


    def positional():
        german = Configuration.german()

        wav2letter = german.load_best_german_model()

        example = german.corpus.examples[0]

        for section in example.sections():
            print(wav2letter.test_and_predict(section))


    def run(use_kenlm=False, language_model_name_extension="",
            index: int = int(sys.argv[1] if len(sys.argv) == 2 else 0)):
        kenlm_extension = ("kenlm" + language_model_name_extension) if use_kenlm else "greedy"

        def logged_german_run(model_name: str, epoch: int) -> LoggedRun:
            return LoggedRun(lambda: Configuration.german().test_german_model(
                model_name, epoch, use_ken_lm=use_kenlm,
                language_model_name_extension=language_model_name_extension),
                             "{}-{}-{}.txt".format(model_name, epoch, kenlm_extension))

        def english_on_english_and_german(model_name: str, epoch: int) -> List[LoggedRun]:
            def test_english_baseline():
                english = Configuration.english()
                # german_frequent_characters, as this model was accidentally trained with these
                # german extra characters will be ignored
                model = english.load_model(model_name, epoch,
                                           use_kenlm=use_kenlm,
                                           language_model_name_extension=language_model_name_extension)
                english.test_model_grouped_by_loaded_corpus_name(model)

            return [LoggedRun(test_english_baseline,
                              "{}-{}-{}-on-English.txt".format(model_name, epoch,
                                                               kenlm_extension)),
                    LoggedRun(lambda: Configuration.german().test_best_english_model(use_kenlm=use_kenlm),
                              "{}-{}-{}.txt".format(model_name, epoch, kenlm_extension))]

        logged_runs = english_on_english_and_german(*Configuration.english_baseline) + [
            logged_german_run(model_name, epoch) for model_name, epoch in
            OldRuns.german_model_names_with_epochs]

        logged_runs[index]()


    # run(use_kenlm=False)  # language_model_name_extension="-incl-trans")

    def validate_to_csv(model_name: str, last_epoch: int,
                        configuration: Configuration = Configuration.german(),
                        epoch_step: int = 5, first_epoch: int = 0,
                        csv_directory: Path = configuration.default_data_directories.test_results_directory) -> List[
        Tuple[int, ExpectationsVsPredictionsInGroupedBatches]]:

        epochs = list(range(first_epoch, last_epoch + 1, epoch_step))[1:]
        log("Testing model {} on epochs {}.".format(model_name, epochs))

        model = configuration.load_model(model_name, last_epoch,
                                         allowed_characters_for_loaded_model=configuration.allowed_characters,
                                         use_kenlm=True)

        def result(epoch: int) -> Tuple[int, ExpectationsVsPredictionsInGroupedBatches]:
            log("Testing epoch {}.".format(epoch))

            model.load_weights(
                allowed_characters_for_loaded_model=configuration.allowed_characters,
                load_model_from_directory=configuration.directories.nets_base_directory / model_name, load_epoch=epoch)

            results = configuration.test_model_grouped_by_loaded_corpus_name(model)
            return epoch, results

        results_with_epochs = [result(epoch) for epoch in epochs]

        csv_file = csv_directory / "{}.csv".format(model_name)
        import csv
        with csv_file.open('w', encoding='utf8') as opened_csv:
            writer = csv.writer(opened_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for epoch, result in results_with_epochs:
                writer.writerow((epoch, result.average_loss, result.average_letter_error_rate,
                                 result.average_word_error_rate, result.average_letter_error_count,
                                 result.average_word_error_count))

        return results_with_epochs


    results = validate_to_csv(Configuration.english_baseline[0], 2000, configuration=Configuration.english(),
                              epoch_step=1000)

    print("Result: {}".format(results))
