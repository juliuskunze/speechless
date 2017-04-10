from pathlib import Path

from corpus import TrainingTestSplit
from english_corpus import LibriSpeechCorpus
from german_corpus import german_corpus
from labeled_example import LabeledExample
from net import Wav2Letter
from net_with_corpus import Wav2LetterWithCorpus
from recording import Recorder
from tools import mkdir, home_directory, timestamp

base_directory = home_directory() / "speechless-data"
tensorboard_log_base_directory = base_directory / "logs"
nets_base_directory = base_directory / "nets"
recording_directory = base_directory / "recordings"

corpus_base_direcory = base_directory / "corpus"
english_corpus_directory = corpus_base_direcory / "English"
german_corpus_directory = corpus_base_direcory / "German"
spectrogram_cache_base_directory = base_directory / "spectrogram-cache"

english_spectrogram_cache_directory = spectrogram_cache_base_directory / "English"
german_spectrogram_cache_directory = spectrogram_cache_base_directory / "German"

batch_size = 2
mel_frequency_count = 128
run_name = timestamp() + "-german-adam-small-learning-rate-complete-95"
wav2letter = Wav2Letter(input_size_per_time_step=mel_frequency_count)
corpus = LibriSpeechCorpus(
    english_corpus_directory, corpus_names=["dev-clean"],
    training_test_split=TrainingTestSplit.overfit(training_example_count=batch_size),
    mel_frequency_count=mel_frequency_count)

wav2letter_with_corpus = Wav2LetterWithCorpus(wav2letter, corpus, batch_size=batch_size,
                                              spectrogram_cache_directory=german_spectrogram_cache_directory)


def load_best_wav2letter_model(mel_frequency_count: int = 128):
    from net import Wav2Letter

    return Wav2Letter(
        input_size_per_time_step=mel_frequency_count,
        load_model_from_directory=Path(nets_base_directory / "20170314-134351-adam-small-learning-rate-complete-95"),
        load_epoch=1689)


def record_plot_and_save() -> LabeledExample:
    from labeled_example_plotter import LabeledExamplePlotter

    print("Wait in silence to begin recording; wait in silence to terminate")
    mkdir(recording_directory)
    name = "recording-{}".format(timestamp())
    example = Recorder().record_to_file(recording_directory / "{}.wav".format(name))
    LabeledExamplePlotter(example).save_spectrogram(recording_directory)

    return example


def predict_recording() -> None:
    wav2letter = load_best_wav2letter_model()

    def predict(sample: LabeledExample) -> str:
        return wav2letter.predict_single(sample.z_normalized_transposed_spectrogram())

    def print_prediction(labeled_example: LabeledExample, description: str = None) -> None:
        print((description if description else labeled_example.id) + ": " + '"{}"'.format(predict(labeled_example)))

    def record_and_print_prediction(description: str = None) -> None:
        print_prediction(record_plot_and_save(), description=description)

    def print_prediction_from_file(file_name: str, description: str = None) -> None:
        print_prediction(LabeledExample(recording_directory / file_name), description=description)

    def print_example_predictions() -> None:
        print("Predictions: ")
        print_prediction_from_file("6930-75918-0000.flac",
                                   description='Sample labeled "concord returned to its place amidst the tents"')
        print_prediction_from_file("recording-20170310-135534.wav", description="Recorded playback of the same sample")
        print_prediction_from_file("recording-20170310-135144.wav",
                                   description="Recording of me saying the same sentence")
        print_prediction_from_file("recording-20170314-224329.wav",
                                   description='Recording of me saying "I just wrote a speech recognizer"')
        print_prediction_from_file("bad-quality-louis.wav",
                                   description="Louis' rerecording of worse quality")

    print_example_predictions()


def summarize_german_corpus():
    corpus = german_corpus(german_corpus_directory)
    corpus.summarize_to_csv(base_directory / "summary.csv")
    print(corpus.summary())


summarize_german_corpus()

# wav2letter_with_corpus.train(batches_per_epoch=10, tensor_board_log_directory=tensorboard_log_base_directory / run_name,
#                              net_directory=nets_base_directory / run_name)
