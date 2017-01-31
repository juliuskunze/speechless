import string
from functools import reduce
from itertools import *
from pathlib import Path
from typing import List

import keras
from keras import backend
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution1D, Lambda
from keras.models import Sequential
from keras.optimizers import SGD, Optimizer
from lazy import lazy
from numpy import *


class Wav2Letter:
    """Speech-recognition network based on wav2letter (https://arxiv.org/pdf/1609.03193v2.pdf)."""

    class InputNames:
        input_batch = "input_batch"
        label_batch = "label_batch"
        prediction_lengths = "prediction_lenghts"
        label_lengths = "label_lenghts"

    def __init__(self,
                 input_size_per_time_step: int,
                 allowed_characters: List[chr] = list(string.ascii_uppercase + " '"),
                 use_raw_wave_input: bool = False,
                 activation: str = "relu",
                 output_activation: str = None,
                 optimizer: Optimizer = SGD(lr=1e-3, momentum=0.9, clipnorm=5)):

        self.output_activation = output_activation
        self.activation = activation
        self.use_raw_wave_input = use_raw_wave_input
        self.input_size_per_time_step = input_size_per_time_step

        self.allowed_characters = allowed_characters
        self.output_grapheme_set_size = len(allowed_characters) + 1
        self.allowed_character_count = len(allowed_characters)
        self.graphemes_by_character = dict((char, index) for index, char in enumerate(allowed_characters))
        self.ctc_blank = self.output_grapheme_set_size - 1  # ctc blank must be last (see Tensorflow's ctcloss documentation)
        self.optimizer = optimizer

    @lazy
    def predictive_net(self) -> Sequential:
        """Returns the part of the net that predicts grapheme probabilities given a spectrogram.
         A loss operation is not contained.
         As described here: https://arxiv.org/pdf/1609.03193v2.pdf
        """

        def convolution(name: str, filter_count: int, filter_length: int, striding: int = 1,
                        activation: str = self.activation,
                        input_dim: int = None) -> Convolution1D:
            return Convolution1D(nb_filter=filter_count, filter_length=filter_length, subsample_length=striding,
                                 activation=activation, name=name, input_dim=input_dim, border_mode="same")

        main_filter_count = 250

        def input_convolutions() -> List[Convolution1D]:
            raw_wave_convolution_if_needed = [
                convolution("wave_conv", filter_count=main_filter_count, filter_length=250, striding=160,
                            input_dim=self.input_size_per_time_step)] if self.use_raw_wave_input else []

            return raw_wave_convolution_if_needed + [
                convolution("striding_conv", filter_count=main_filter_count, filter_length=48, striding=2,
                            input_dim=None if self.use_raw_wave_input else self.input_size_per_time_step)]

        def inner_convolutions() -> List[Convolution1D]:
            return [convolution("inner_conv_{}".format(i), filter_count=main_filter_count, filter_length=7) for i in
                    range(1, 8)]

        def output_convolutions() -> List[Convolution1D]:
            out_filter_count = 2000
            return [
                convolution("big_conv_1", filter_count=out_filter_count, filter_length=32),
                convolution("big_conv_2", filter_count=out_filter_count, filter_length=1),
                convolution("output_conv", filter_count=self.output_grapheme_set_size, filter_length=1,
                            activation=self.output_activation)
            ]

        return Sequential(
            input_convolutions() +
            inner_convolutions() +
            output_convolutions())

    @lazy
    def input_to_prediction_length_ratio(self):
        """Returns which factor shorter the output is compared to the input caused by striding."""
        return reduce(lambda x, y: x * y, [layer.subsample_length for layer in self.predictive_net.layers], 1)

    def prediction_batch(self, input_batch: ndarray) -> ndarray:
        """Predicts a grapheme probability batch given a spectrogram batch, employing the predictive network."""
        return backend.function(self.predictive_net.inputs, self.predictive_net.outputs)([input_batch])[0]

    @lazy
    def loss_net(self) -> Model:
        """Returns the network that yields a loss given both input spectrograms and labels. Used for training."""
        input_batch = Input(name=Wav2Letter.InputNames.input_batch, batch_shape=self.predictive_net.input_shape)
        label_batch = Input(name=Wav2Letter.InputNames.label_batch, shape=(None,))
        prediction_lengths = Input(name=Wav2Letter.InputNames.prediction_lengths, shape=(1,), dtype='int64')
        label_lengths = Input(name=Wav2Letter.InputNames.label_lengths, shape=(1,), dtype='int64')

        # Since Keras doesn't currently support loss functions with extra parameters,
        # we define a custom lambda layer yielding one single real-valued CTC loss given the grapheme probabilities:
        loss_layer = Lambda(Wav2Letter._ctc_lambda, name='ctc_loss', output_shape=(1,))

        # This loss layer is placed atop the predictive network and provided with additional arguments,
        # namely the label batch and prediction/label sequence lengths:
        loss = loss_layer([self.predictive_net(input_batch), label_batch, prediction_lengths, label_lengths])

        loss_net = Model(input=[input_batch, label_batch, prediction_lengths, label_lengths], output=[loss])
        # Since loss is already calculated in the last layer of the net, we just pass through the results here.
        # The loss dummy labels have to be given to satify the Keras API.
        loss_net.compile(loss=lambda dummy_labels, ctc_loss: ctc_loss, optimizer=self.optimizer)
        return loss_net

    # No type hints here because the annotations cause an error in the Keras library:
    @staticmethod
    def _ctc_lambda(args):
        prediction_batch, label_batch, prediction_lengths, label_lengths = args
        return backend.ctc_batch_cost(y_true=label_batch, y_pred=prediction_batch, input_length=prediction_lengths,
                                      label_length=label_lengths)

    def predict(self, input_batch: ndarray) -> List[str]:
        return self.decode_prediction_batch(self.prediction_batch(input_batch))

    def train(self, spectrograms: List[ndarray], labels: List[str], tensor_board_log_directory: Path):
        training_input_dictionary = self._training_input_dictionary(spectrograms, labels)

        input_batch = training_input_dictionary[Wav2Letter.InputNames.input_batch]

        def print_expectations_vs_prediction():
            print("\n".join(
                'Expected:  "{}"\nPredicted: "{}"'.format(expected.lower(), predicted.lower()) for expected, predicted
                in zip(labels, self.predict(input_batch))))

        print_expectations_vs_prediction()

        batch_size = len(spectrograms)
        dummy_labels_for_dummy_loss_function = zeros((batch_size,))

        def generate_data():
            while True:
                yield (training_input_dictionary, dummy_labels_for_dummy_loss_function)

        self.loss_net.fit_generator(generate_data(), samples_per_epoch=20, nb_epoch=500000000000000,
                                    callbacks=self.create_callbacks(
                                        test_input_batch=input_batch,
                                        tensor_board_log_directory=tensor_board_log_directory))

        print_expectations_vs_prediction()

    def create_callbacks(self, test_input_batch: ndarray, tensor_board_log_directory: Path):
        class ComparisonCallback(keras.callbacks.Callback):
            def on_epoch_end(self2, epoch, logs=()):
                if epoch % 100 == 0:
                    print(self.predict(input_batch=test_input_batch))

        tensorboard = keras.callbacks.TensorBoard(log_dir=str(tensor_board_log_directory), write_images=True)
        return [tensorboard, ComparisonCallback()]

    def decode_prediction_batch(self, prediction_batch: ndarray) -> List[str]:
        # TODO use beam search with a language model instead of best path.
        return [self.decode_graphemes(list(argmax(prediction_batch[i], 1))) for i in
                range(prediction_batch.shape[0])]

    def decode_graphemes(self, graphemes: List[int]) -> str:
        grouped_graphemes = [k for k, g in groupby(graphemes)]
        return self.decode_grouped_graphemes(grouped_graphemes)

    def decode_grouped_graphemes(self, grouped_graphemes: List[int]) -> str:
        return "".join([self.decode_grapheme(grapheme) for grapheme in grouped_graphemes])

    def encode(self, label: str) -> List[int]:
        return [self.encode_char(c) for c in label]

    def decode_grapheme(self, grapheme: int) -> str:
        if grapheme in range(self.allowed_character_count):
            return self.allowed_characters[grapheme]
        elif grapheme == self.ctc_blank:
            return ""
        else:
            raise ValueError("Unexpected grapheme: '{}'".format(grapheme))

    def encode_char(self, label_char: chr) -> int:
        try:
            return self.graphemes_by_character[label_char]
        except:
            raise ValueError("Unexpected char: '{}'".format(label_char))

    def _training_input_dictionary(self, spectrograms: List[ndarray], labels: List[str]) -> dict:
        """
        :param spectrograms: In shape (time, channels)
        :param labels:
        :return:
        """
        assert (len(labels) == len(spectrograms))

        batch_size = len(spectrograms)

        input_size_per_time_step = spectrograms[0].shape[1]

        input_lengths = [spectrogram.shape[0] for spectrogram in spectrograms]
        prediction_lengths = [s / self.input_to_prediction_length_ratio for s in input_lengths]
        input_batch = zeros((batch_size, max(input_lengths), input_size_per_time_step))
        for index, spectrogram in enumerate(spectrograms):
            input_batch[index, :spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram

        label_lengths = [len(label) for label in labels]
        label_batch = -ones((batch_size, max(label_lengths)))
        for index, label in enumerate(labels):
            label_batch[index, :len(label)] = array(self.encode(label))

        print(input_batch.shape, label_batch.shape, input_lengths, label_lengths)
        return {
            Wav2Letter.InputNames.input_batch: input_batch,
            Wav2Letter.InputNames.label_batch: label_batch,
            Wav2Letter.InputNames.prediction_lengths: reshape(array(prediction_lengths), (batch_size, 1)),
            Wav2Letter.InputNames.label_lengths: reshape(array(label_lengths), (batch_size, 1))
        }
