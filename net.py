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
from keras.optimizers import SGD
from lazy import lazy
from numpy import *


class Wav2Letter:
    allowed_characters = list(string.ascii_uppercase + " '")
    allowed_character_count = len(allowed_characters)
    graphemes_by_character = dict((char, index) for index, char in enumerate(allowed_characters))

    ctc_grapheme_set_size = len(allowed_characters) + 1
    ctc_blank = ctc_grapheme_set_size - 1  # ctc blank must be last (see Tensorflow's ctcloss documentation)

    class InputNames:
        input_batch = "input_batch"
        label_batch = "label_batch"
        prediction_lengths = "prediction_lenghts"
        label_lengths = "label_lenghts"

    def __init__(self,
                 input_size_per_time_step: int,
                 output_grapheme_set_size: int = ctc_grapheme_set_size,
                 use_raw_wave_input: bool = False,
                 activation: str = "relu",
                 output_activation: str = None):

        self.output_activation = output_activation
        self.activation = activation
        self.use_raw_wave_input = use_raw_wave_input
        self.output_grapheme_set_size = output_grapheme_set_size
        self.input_size_per_time_step = input_size_per_time_step

    @lazy
    def prediction_net(self) -> Sequential:
        """As described in https://arxiv.org/pdf/1609.03193v2.pdf.
         This function returns the net architecture of the predictive
         part of the wav2letter architecture (without a loss operation).
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
        """ Returns which factor shorter the output is compared to the input caused by striding."""
        return reduce(lambda x, y: x * y, [layer.subsample_length for layer in self.prediction_net.layers], 1)

    def prediction_batch(self, input_batch: ndarray) -> ndarray:
        return backend.function(self.prediction_net.inputs, self.prediction_net.outputs)([input_batch])[0]

    @lazy
    def net_with_ctc_loss(self) -> Model:
        input_batch = Input(name=Wav2Letter.InputNames.input_batch, batch_shape=self.prediction_net.input_shape)
        # TODO shape=[None], this may fail, replace by "absolute max string length"?
        label_batch = Input(name=Wav2Letter.InputNames.label_batch, shape=[None])
        prediction_lengths = Input(name=Wav2Letter.InputNames.prediction_lengths, shape=[1], dtype='int64')
        label_lengths = Input(name=Wav2Letter.InputNames.label_lengths, shape=[1], dtype='int64')
        prediction_batch = self.prediction_net(input_batch)

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        # the actual loss calc occurs here despite it not being an internal Keras loss function
        losses = Lambda(Wav2Letter._ctc_lambda, output_shape=(1,), name='ctc')(
            [prediction_batch, label_batch, prediction_lengths, label_lengths])
        return Model(input=[input_batch, label_batch, prediction_lengths, label_lengths], output=[losses])

    # no type hints here because the annotations cause an error in the Keras library
    @staticmethod
    def _ctc_lambda(args):
        prediction_batch, label_batch, prediction_lengths, label_lengths = args
        # TODO check: the keras implementation takes the logarithm of the predictions (see keras.backend.ctc_batch_cost)
        return backend.ctc_batch_cost(y_true=label_batch, y_pred=prediction_batch, input_length=prediction_lengths,
                                      label_length=label_lengths)

    def _compile(self):
        # TODO adjust parameters, "clipnorm seems to speeds up convergence"
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        # the loss calc occurs in the ctc layer defined in the _wav2letter_net_with_ctc_loss net,
        # we just pass through the results here in this dummy loss function:
        self.net_with_ctc_loss.compile(loss=lambda dummy_labels, losses_from_ctc: losses_from_ctc, optimizer=sgd)

    def predict(self, input_batch: ndarray) -> List[str]:
        return self.decode_prediction_batch(self.prediction_batch(input_batch))

    def train(self, spectrograms: List[ndarray], labels: List[str], tensor_board_log_directory: Path):
        self._compile()

        training_input_dictionary = self._training_input_dictionary(spectrograms, labels)

        input_batch = training_input_dictionary[Wav2Letter.InputNames.input_batch]

        def print_decoded():
            print(self.predict(input_batch))

        print_decoded()

        batch_size = len(spectrograms)
        dummy_labels_for_dummy_loss_function = zeros((batch_size,))

        self.net_with_ctc_loss.fit(
            x=training_input_dictionary, y=dummy_labels_for_dummy_loss_function, batch_size=10, nb_epoch=100,
            callbacks=[keras.callbacks.TensorBoard(log_dir=str(tensor_board_log_directory), write_images=True)])

        print_decoded()

    @classmethod
    def decode_prediction_batch(cls, prediction_batch: ndarray) -> List[str]:
        # TODO use beam search with a language model instead of best path.
        return [Wav2Letter.decode_graphemes(list(argmax(prediction_batch[i], 1))) for i in
                range(prediction_batch.shape[0])]

    @classmethod
    def decode_graphemes(cls, graphemes: List[int]) -> str:
        grouped_graphemes = [k for k, g in groupby(graphemes)]
        return cls.decode_grouped_graphemes(grouped_graphemes)

    @classmethod
    def decode_grouped_graphemes(cls, grouped_graphemes: List[int]) -> str:
        return "".join([cls.decode_grapheme(grapheme) for grapheme in grouped_graphemes])

    @classmethod
    def encode(cls, label: str) -> List[int]:
        return [cls.encode_char(c) for c in label]

    @classmethod
    def decode_grapheme(cls, grapheme: int) -> str:
        if grapheme in range(Wav2Letter.allowed_character_count):
            return cls.allowed_characters[grapheme]
        elif grapheme == cls.ctc_blank:
            return ""
        else:
            raise ValueError("Unexpected grapheme: '{}'".format(grapheme))

    @classmethod
    def encode_char(cls, label_char: chr) -> int:
        try:
            return cls.graphemes_by_character[label_char]
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
            label_batch[index, :len(label)] = array(Wav2Letter.encode(label))

        print(input_batch.shape, label_batch.shape, input_lengths, label_lengths)
        return {
            Wav2Letter.InputNames.input_batch: input_batch,
            Wav2Letter.InputNames.label_batch: label_batch,
            Wav2Letter.InputNames.prediction_lengths: reshape(array(prediction_lengths), (batch_size, 1)),
            Wav2Letter.InputNames.label_lengths: reshape(array(label_lengths), (batch_size, 1))
        }
