from socket import gethostname

from typing import Tuple

from speechless import configuration
from speechless.configuration import Configuration
from speechless.grapheme_enconding import german_frequent_characters


def test_german_model(model: Tuple[str, int], use_ken_lm=True):
    load_name, load_epoch = model
    german = Configuration.german()

    german.test_model(german.load_model(
        load_name=load_name,
        load_epoch=load_epoch,
        allowed_characters_for_loaded_model=german_frequent_characters,
        use_ken_lm=use_ken_lm))


transfer1 = ("20170415-092748-adam-small-learning-rate-transfer-to-German-freeze-10", 1778)
transfer3 = ("20170418-120145-adam-small-learning-rate-transfer-to-German-freeze-8", 1759)
german_from_scratch = ("20170415-001150-adam-small-learning-rate-complete-training-German", 443)

if __name__ == '__main__':
    if gethostname() == "ketos":
        ketos_spectrogram_cache_base_directory = configuration.base_directory / "ketos-spectrogram-cache"
        ketos_kenlm_base_directory = configuration.base_directory / "ketos-kenlm"

        print("Running on ketos, using spectrogram cache base directory {} and kenlm base directory {}".format(
            ketos_spectrogram_cache_base_directory, ketos_kenlm_base_directory))
        configuration.spectrogram_cache_base_directory = ketos_spectrogram_cache_base_directory
        configuration.kenlm_base_directory = ketos_kenlm_base_directory

    # Configuration.german(from_cached=False).summarize_and_save_corpus()

    # Configuration.german().fill_cache(repair_incorrect=True)

    # Configuration.german().test_best_model()

    # Configuration.english().summarize_and_save_corpus()

    # Configuration.german().train()

    # net = Configuration.english().load_best_english_model().predictive_net

    # Configuration.german().train_transfer_from_best_english_model(trainable_layer_count=4)

    # Configuration.english().save_corpus()

    test_german_model(german_from_scratch, use_ken_lm=True)
