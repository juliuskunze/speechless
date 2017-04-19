from speechless.configuration import Configuration

if __name__ == '__main__':
    pass
    # Configuration.german(from_cached=False).summarize_and_save_corpus()

    # Configuration.german().fill_cache(repair_incorrect=True)

    # Configuration.german().test_best_model()

    # Configuration.english().summarize_and_save_corpus()

    # Configuration.german().train()

    # net = Configuration.english().load_best_english_model().predictive_net

    # Configuration.german().train_transfer_from_best_english_model(trainable_layer_count=3,
    #                                                               reinitialize_trainable_loaded_layers=True)

    Configuration.english().test_best_model(use_ken_lm=True)
