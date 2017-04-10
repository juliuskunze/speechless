from pathlib import Path

from corpus import Corpus, LabeledSpectrogramBatchGenerator
from net import Wav2Letter


class Wav2LetterWithCorpus:
    def __init__(self, wav2letter: Wav2Letter, corpus: Corpus, spectrogram_cache_directory: Path, batch_size: int):
        self.batch_size = batch_size
        self.wav2letter = wav2letter
        self.corpus = corpus
        self.batch_generator = LabeledSpectrogramBatchGenerator(corpus=self.corpus,
                                                                spectrogram_cache_directory=spectrogram_cache_directory,
                                                                batch_size=self.batch_size)

    def train(self, tensor_board_log_directory: Path, net_directory: Path, batches_per_epoch: int = 100) -> None:
        self.wav2letter.train(self.batch_generator.training_batches(),
                              tensor_board_log_directory=tensor_board_log_directory,
                              net_directory=net_directory,
                              test_labeled_spectrogram_batch=self.batch_generator.preview_batch(),
                              samples_per_epoch=self.batch_generator.batch_size * batches_per_epoch)

    def test(self):
        print(self.wav2letter.loss(self.batch_generator.test_batches()))
