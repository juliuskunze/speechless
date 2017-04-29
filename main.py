import sys
from socket import gethostname

from speechless import configuration
from speechless.configuration import Configuration, LoggedRun
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
    if gethostname() == "ketos":
        ketos_spectrogram_cache_base_directory = configuration.base_directory / "ketos-spectrogram-cache"
        ketos_kenlm_base_directory = configuration.base_directory / "ketos-kenlm"

        log("Running on ketos, using spectrogram cache base directory {} and kenlm base directory {}".format(
            ketos_spectrogram_cache_base_directory, ketos_kenlm_base_directory))
        configuration.spectrogram_cache_base_directory = ketos_spectrogram_cache_base_directory
        configuration.kenlm_base_directory = ketos_kenlm_base_directory
    else:
        restrict_gpu_memory()


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

    Configuration.german().train_transfer_from_best_english_model(frozen_layer_count=0)


    def test_german(use_kenlm=False, language_model_name_extension="",
                    index: int = int(sys.argv[1])):
        kenlm_extension = ("kenlm" + language_model_name_extension) if use_kenlm else "greedy"

        def logged_german_run(model_name: str, epoch: int) -> LoggedRun:
            return LoggedRun(lambda: Configuration.german().test_german_model(
                model_name, epoch, use_ken_lm=use_kenlm,
                language_model_name_extension=language_model_name_extension),
                             "{}-{}-{}.txt".format(model_name, epoch, kenlm_extension))

        def test_english_baseline():
            english = Configuration.english()
            # german_frequent_characters, as this model was accidentally trained with these
            # german extra characters will be ignored
            model = english.load_model(Configuration.english_baseline[0],
                                       Configuration.english_baseline[1],
                                       use_kenlm=use_kenlm,
                                       language_model_name_extension=language_model_name_extension)
            english.test_model(model)

        english_on_english = LoggedRun(test_english_baseline,
                                       "{}-{}-{}-on-English.txt".format(Configuration.english_baseline[0],
                                                                        Configuration.english_baseline[1],
                                                                        kenlm_extension))

        baseline_run = LoggedRun(lambda: Configuration.german().test_best_english_model(use_kenlm=use_kenlm),
                                 "{}-{}-{}.txt".format(Configuration.english_baseline[0],
                                                       Configuration.english_baseline[1],
                                                       kenlm_extension))
        logged_runs = [english_on_english, baseline_run] + [
            logged_german_run(model_name, epoch) for model_name, epoch in
            Configuration.german_model_names_with_epochs]

        logged_runs[index]()

        # test_german(use_kenlm=False, language_model_name_extension="-incl-trans")
