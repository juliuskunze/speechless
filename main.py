import sys
from socket import gethostname

from speechless import configuration
from speechless.configuration import Configuration, LoggedRun
from speechless.tools import log

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

    Configuration.german(sampled_training_example_count_when_loading_from_cached=50000).train_from_beginning()

    # net = Configuration.english().load_best_english_model().predictive_net

    # Configuration.german(sampled_training_example_count_when_loading_from_cached=10000). \
    #    train_transfer_from_best_english_model(frozen_layer_count=8)

    # Configuration.english().save_corpus()

    def test():
        index = int(sys.argv[1])

        german = Configuration.german()
        use_kenlm = False
        use_old_language_model = False
        kenlm_extension = ("kenlm-old" if use_old_language_model else "kenlm") if use_kenlm else "greedy"

        def logged_german_run(model_name: str, epoch: int) -> LoggedRun:
            return LoggedRun(lambda: german.test_german_model(model_name, epoch, use_ken_lm=use_kenlm,
                                                              use_old_language_model=use_old_language_model),
                             "{}-{}-{}.txt".format(model_name, epoch, kenlm_extension))

        logged_runs = [LoggedRun(lambda: german.test_best_english_model(use_kenlm=use_kenlm),
                                 "{}-{}-{}.txt".format(Configuration.freeze11[0], Configuration.freeze11[1],
                                                       kenlm_extension))] + [
                          logged_german_run(model_name, epoch) for model_name, epoch in
                          Configuration.german_model_names_with_epochs]

        logged_runs[index]()
