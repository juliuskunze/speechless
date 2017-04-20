import logging
from socket import gethostname

from typing import Callable

from speechless import configuration
from speechless.configuration import Configuration
from speechless.tools import log, logger, mkdir, write_text


class LoggedRun:
    def __init__(self, action: Callable[[], None], name: str):
        self.name = name
        self.action = action

    def __call__(self, *args, **kwargs):
        results_directory = configuration.base_directory / "test-results"
        mkdir(results_directory)
        result_file = results_directory / self.name
        write_text(result_file, "")
        handler = logging.FileHandler(str(result_file))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            self.action()
        finally:
            logger.removeHandler(handler)


if __name__ == '__main__':
    if gethostname() == "ketos":
        ketos_spectrogram_cache_base_directory = configuration.base_directory / "ketos-spectrogram-cache"
        ketos_kenlm_base_directory = configuration.base_directory / "ketos-kenlm"

        log("Running on ketos, using spectrogram cache base directory {} and kenlm base directory {}".format(
            ketos_spectrogram_cache_base_directory, ketos_kenlm_base_directory))
        configuration.spectrogram_cache_base_directory = ketos_spectrogram_cache_base_directory
        configuration.kenlm_base_directory = ketos_kenlm_base_directory

    # Configuration.german(from_cached=False).summarize_and_save_corpus()

    # Configuration.german().fill_cache(repair_incorrect=True)

    # Configuration.german().test_best_model()

    # Configuration.english().summarize_and_save_corpus()

    # Configuration.german().train()

    # net = Configuration.english().load_best_english_model().predictive_net

    # Configuration.german(sampled_training_example_count_when_loading_from_cached=50000). \
    #    train_transfer_from_best_english_model(frozen_layer_count=8)

    # Configuration.english().save_corpus()

    german = Configuration.german()
    use_kenlm = False
    kenlm_extension = "kenlm" if use_kenlm else "greedy"
    logged_runs = [LoggedRun(lambda: german.test_german_model(logged_run, use_ken_lm=use_kenlm),
                             "{}-{}.txt".format(logged_run[0], kenlm_extension))
                   for logged_run in Configuration.german_model_names_with_epochs]

    for logged_run in logged_runs:
        logged_run()
