from pathlib import Path
from time import strftime

from corpus_provider import CorpusProvider
from german_corpus_provider import GermanCorpusProvider, german_corpus_definitions_sorted_by_size
from labeled_example import LabeledExample
from net import Wav2Letter
from recording import Recorder
from spectrogram_batch import LabeledSpectrogramBatchGenerator
from tools import mkdir, home_directory

base_directory = home_directory() / "speechless-data"
base_spectrogram_directory = base_directory / "spectrograms"

mkdir(base_spectrogram_directory)
tensorboard_log_base_directory = base_directory / "logs"
nets_base_directory = base_directory / "nets"
recording_directory = base_directory / "recordings"
corpus_directory = base_directory / "corpus"
german_corpus_directory = base_directory / "german-corpus"
spectrogram_cache_directory = base_directory / "spectrogram-cache" / "mel"


def timestamp() -> str:
    return strftime("%Y%m%d-%H%M%S")


def train_wav2letter(mel_frequency_count: int = 128) -> None:
    labeled_spectrogram_batch_generator = batch_generator(mel_frequency_count=mel_frequency_count)

    wav2letter = load_best_wav2letter_model(mel_frequency_count=mel_frequency_count)

    run_name = timestamp() + "-adam-small-learning-rate-complete-95"

    wav2letter.train(labeled_spectrogram_batch_generator.as_training_batches(),
                     tensor_board_log_directory=tensorboard_log_base_directory / run_name,
                     net_directory=nets_base_directory / run_name,
                     test_labeled_spectrogram_batch=labeled_spectrogram_batch_generator.preview_batch(),
                     samples_per_epoch=labeled_spectrogram_batch_generator.batch_size * 100)


def batch_generator(is_training: bool = True, mel_frequency_count: int = 128) -> LabeledSpectrogramBatchGenerator:
    corpus = CorpusProvider(corpus_directory, mel_frequency_count=mel_frequency_count)
    split_index = int(len(corpus.examples) * .95)
    examples = corpus.examples[:split_index] if is_training else corpus.examples[split_index:]
    return LabeledSpectrogramBatchGenerator(examples=examples, spectrogram_cache_directory=spectrogram_cache_directory)


def record() -> LabeledExample:
    print("Wait in silence to begin recording; wait in silence to terminate")
    mkdir(recording_directory)
    name = "recording-{}".format(timestamp())
    example = Recorder().record_to_file(recording_directory / "{}.wav".format(name))
    example.save_spectrogram(recording_directory)

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
        print_prediction(LabeledExample.from_file(recording_directory / file_name), description=description)

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


def validate_best_model():
    wav2_letter = load_best_wav2letter_model()

    generator = batch_generator(is_training=False)

    print(wav2_letter.loss(generator.as_validation_batches()))


def load_best_wav2letter_model(mel_frequency_count: int = 128):
    return Wav2Letter(
        input_size_per_time_step=mel_frequency_count,
        load_model_from_directory=Path(nets_base_directory / "20170314-134351-adam-small-learning-rate-complete-95"),
        load_epoch=1689)


def summarize_german_corpus() -> None:
    for corpus_name in german_corpus_definitions_sorted_by_size:
        print(GermanCorpusProvider(german_corpus_directory, corpus_name).summary())


summarize_german_corpus()
