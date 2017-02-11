import string
from functools import reduce
from itertools import *
from os import makedirs
from pathlib import Path
from typing import List, Callable

import keras
import tensorflow as tf
from keras import backend
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution1D, Lambda
from keras.models import Sequential
from keras.optimizers import SGD, Optimizer
from lazy import lazy
from numpy import *

from grapheme_enconding import CtcGraphemeEncoding


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
        self.grapheme_encoding = CtcGraphemeEncoding(allowed_characters=allowed_characters)
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
                convolution("output_conv", filter_count=self.grapheme_encoding.grapheme_set_size, filter_length=1,
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
        return Wav2Letter.ctc_batch_cost_no_log(y_true=label_batch, y_pred=prediction_batch,
                                                input_length=prediction_lengths, label_length=label_lengths)

    @staticmethod
    def ctc_batch_cost_no_log(y_true, y_pred, input_length, label_length):
        """Copied from keras.backend.ctc_batch_cost, with the difference shown below."""
        label_length = tf.to_int32(tf.squeeze(label_length))
        input_length = tf.to_int32(tf.squeeze(input_length))
        sparse_labels = tf.to_int32(keras.backend.ctc_label_dense_to_sparse(y_true, label_length))

        # Difference here: no tf.log(... + 1e-8)
        y_pred = tf.transpose(y_pred, perm=[1, 0, 2])

        return tf.expand_dims(keras.backend.ctc.ctc_loss(inputs=y_pred,
                                                         labels=sparse_labels,
                                                         sequence_length=input_length), 1)

    def predict(self, spectrograms: List[ndarray]) -> List[str]:
        input_batch, prediction_lengths = self._input_batch_and_prediction_lengths(spectrograms)

        return self.grapheme_encoding.decode_prediction_batch(self.prediction_batch(input_batch),
                                                              prediction_lengths=prediction_lengths)

    def train(self, spectrograms: List[ndarray], labels: List[str], tensor_board_log_directory: Path,
              net_directory: Path):
        # not Path.mkdir() for compatibility with Python 3.4
        makedirs(str(net_directory), exist_ok=True)

        def print_expectations_vs_prediction():
            print("\n\n".join(
                'Expected:  "{}"\nPredicted: "{}"'.format(expected.lower(), predicted.lower()) for expected, predicted
                in zip(labels, self.predict(spectrograms=spectrograms))))

        print_expectations_vs_prediction()

        batch_size = len(spectrograms)
        dummy_labels_for_dummy_loss_function = zeros((batch_size,))

        training_input_dictionary = self._training_input_dictionary(spectrograms, labels)

        def generate_data():
            while True:
                yield (training_input_dictionary, dummy_labels_for_dummy_loss_function)

        self.loss_net.fit_generator(generate_data(), samples_per_epoch=20, nb_epoch=100000000,
                                    callbacks=self.create_callbacks(
                                        callback=print_expectations_vs_prediction,
                                        tensor_board_log_directory=tensor_board_log_directory,
                                        net_directory=net_directory))

    def create_callbacks(self, callback: Callable[[], None], tensor_board_log_directory: Path, net_directory: Path,
                         callback_step: int = 100, save_step: int = 10000) -> List[keras.callbacks.Callback]:
        class CustomCallback(keras.callbacks.Callback):
            def on_epoch_end(self_callback, epoch, logs=()):
                if epoch % callback_step == 0:
                    callback()

                if epoch % save_step == 0:
                    self.predictive_net.save(str(net_directory / "predictive-epoch{}.kerasnet".format(epoch)))
                    self.loss_net.save(str(net_directory / "loss-epoch{}.kerasnet".format(epoch)))

        tensorboard = keras.callbacks.TensorBoard(log_dir=str(tensor_board_log_directory), write_images=True)
        return [tensorboard, CustomCallback()]

    def _input_batch_and_prediction_lengths(self, spectrograms: List[ndarray]):
        batch_size = len(spectrograms)
        input_size_per_time_step = spectrograms[0].shape[1]
        input_lengths = [spectrogram.shape[0] for spectrogram in spectrograms]
        prediction_lengths = [s // self.input_to_prediction_length_ratio for s in input_lengths]
        input_batch = zeros((batch_size, max(input_lengths), input_size_per_time_step))
        for index, spectrogram in enumerate(spectrograms):
            input_batch[index, :spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram

        return input_batch, prediction_lengths

    def _training_input_dictionary(self, spectrograms: List[ndarray], labels: List[str]) -> dict:
        """
        :param spectrograms: In shape (time, channels)
        :param labels:
        :return:
        """
        assert (len(labels) == len(spectrograms))

        input_batch, prediction_lengths = self._input_batch_and_prediction_lengths(spectrograms)

        return {
            Wav2Letter.InputNames.input_batch: input_batch,
            Wav2Letter.InputNames.prediction_lengths: reshape(array(prediction_lengths), (len(spectrograms), 1)),
            Wav2Letter.InputNames.label_batch: self.grapheme_encoding.encode_label_batch(labels),
            Wav2Letter.InputNames.label_lengths: reshape(array([len(label) for label in labels]), (len(labels), 1))
        }
