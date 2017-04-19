from speechless.configuration import Configuration
from speechless.grapheme_enconding import german_frequent_characters

if __name__ == '__main__':
    # Configuration.german(from_cached=False).summarize_and_save_corpus()

    # Configuration.german().fill_cache(repair_incorrect=True)

    # Configuration.german().test_best_model()

    # Configuration.english().summarize_and_save_corpus()

    # Configuration.german().train()

    # net = Configuration.english().load_best_english_model().predictive_net

    # Configuration.german().train_transfer_from_best_english_model(trainable_layer_count=3,
    #                                                               reinitialize_trainable_loaded_layers=True)

    Configuration.english().save_corpus()


    def test_transfer1():
        german = Configuration.german()

        german.test_model(german.load_model(
            load_name="20170415-092748-adam-small-learning-rate-transfer-to-German-freeze-10",
            load_epoch=1778,
            allowed_characters_for_loaded_model=german_frequent_characters))
