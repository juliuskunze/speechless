from pathlib import Path
from time import strftime

from corpus_provider import CorpusProvider
from german_corpus_provider import german_corpus_providers
from labeled_example import LabeledExample
from recording import Recorder
from spectrogram_batch import LabeledSpectrogramBatchGenerator
from tools import mkdir, home_directory

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


def timestamp() -> str:
    return strftime("%Y%m%d-%H%M%S")


def train_wav2letter(mel_frequency_count: int = 128, epoch_size: int = 100) -> None:
    from net import Wav2Letter

    labeled_spectrogram_batch_generator = batch_generator(mel_frequency_count=mel_frequency_count)

    wav2letter = Wav2Letter(input_size_per_time_step=mel_frequency_count,
                            use_asg=True)

    run_name = timestamp() + "-german-adam-small-learning-rate-complete-95"

    wav2letter.train(labeled_spectrogram_batch_generator.as_training_batches(),
                     tensor_board_log_directory=tensorboard_log_base_directory / run_name,
                     net_directory=nets_base_directory / run_name,
                     test_labeled_spectrogram_batch=labeled_spectrogram_batch_generator.preview_batch(),
                     samples_per_epoch=labeled_spectrogram_batch_generator.batch_size * epoch_size)


def batch_generator(is_training: bool = True, mel_frequency_count: int = 128) -> LabeledSpectrogramBatchGenerator:
    # TODO use specified mel frequency count
    corpus = CorpusProvider(english_corpus_directory, corpus_names=["dev-clean"])
    # TODO fix this, sample randomly:
    split_index = int(len(corpus.examples) * .95)

    tiny_batch_size = 2
    examples = corpus.examples[:split_index][:tiny_batch_size] if is_training else corpus.examples[split_index:]
    return LabeledSpectrogramBatchGenerator(examples=examples,
                                            spectrogram_cache_directory=german_spectrogram_cache_directory,
                                            batch_size=tiny_batch_size)


def record() -> LabeledExample:
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
        print_prediction(record(), description=description)

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


def validate_best_model() -> None:
    wav2_letter = load_best_wav2letter_model()

    generator = batch_generator(is_training=False)

    print(wav2_letter.loss(generator.as_validation_batches()))


def load_best_wav2letter_model(mel_frequency_count: int = 128):
    from net import Wav2Letter

    return Wav2Letter(
        input_size_per_time_step=mel_frequency_count,
        load_model_from_directory=Path(nets_base_directory / "20170314-134351-adam-small-learning-rate-complete-95"),
        load_epoch=1689)


def summarize_german_corpus() -> None:
    import csv
    with (base_directory / "summary.csv").open('w', encoding='utf8') as csv_summary_file:
        writer = csv.writer(csv_summary_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for corpus_provider in german_corpus_providers(german_corpus_directory):
            print(corpus_provider.summary())
            writer.writerow(corpus_provider.csv_row())


train_wav2letter(epoch_size=10)
# summarize_german_corpus()
