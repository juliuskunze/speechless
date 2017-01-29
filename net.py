import string
from itertools import *
from pathlib import Path
from typing import List, Dict

import keras
from keras import backend
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution1D, Lambda
from keras.models import Sequential
from keras.optimizers import SGD
from numpy import *


def wav2letter_net(
        input_size_per_time_step: int,
        output_grapheme_set_size: int,
        use_raw_wave_input: bool = False,
        activation: str = "relu",
        border_mode: str = "same",
        output_activation: str = None
) -> Sequential:
    """As described in https://arxiv.org/pdf/1609.03193v2.pdf.
     This function returns the net architecture of the predictive
     part of the wav2letter archicture (without a loss operation).
    """

    def convolution(name: str, filter_count: int, filter_length: int, striding: int = 1, activation: str = activation,
                    input_dim: int = None) -> Convolution1D:
        return Convolution1D(nb_filter=filter_count, filter_length=filter_length, subsample_length=striding,
                             activation=activation, name=name, input_dim=input_dim, border_mode=border_mode)

    main_filter_count = 250

    def input_convolutions() -> List[Convolution1D]:
        raw_wave_convolution_if_needed = [
            convolution("wave_conv", filter_count=main_filter_count, filter_length=250, striding=160,
                        input_dim=input_size_per_time_step)] if use_raw_wave_input else []

        return raw_wave_convolution_if_needed + [
            convolution("striding_conv", filter_count=main_filter_count, filter_length=48, striding=2,
                        input_dim=None if use_raw_wave_input else input_size_per_time_step)]

    def inner_convolutions() -> List[Convolution1D]:
        return [convolution("inner_conv_{}".format(i), filter_count=main_filter_count, filter_length=7) for i in
                range(1, 8)]

    def output_convolutions() -> List[Convolution1D]:
        out_filter_count = 2000
        return [
            convolution("big_conv_1", filter_count=out_filter_count, filter_length=32),
            convolution("big_conv_2", filter_count=out_filter_count, filter_length=1),
            convolution("output_conv", filter_count=output_grapheme_set_size, filter_length=1,
                        activation=output_activation)
        ]

    return Sequential(
        input_convolutions() +
        inner_convolutions() +
        output_convolutions())


def _wav2letter_net_with_ctc_loss(model: Sequential) -> Model:
    input_batch = Input(name=InputNames.input_batch, batch_shape=model.input_shape)
    # TODO shape=[None], this may fail, replace by "absolute max string length"?
    label_batch = Input(name=InputNames.label_batch, shape=[None])
    prediction_lengths = Input(name=InputNames.prediction_lengths, shape=[1], dtype='int64')
    label_lengths = Input(name=InputNames.label_lengths, shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    prediction_batch = model(input_batch)

    losses = Lambda(_ctc_lambda, output_shape=(1,), name='ctc')(
        [prediction_batch, label_batch, prediction_lengths, label_lengths])
    return Model(input=[input_batch, label_batch, prediction_lengths, label_lengths], output=[losses])


# the actual loss calc occurs here despite it not being an internal Keras loss function
# no type hints here because the annotations cause an error in the Keras library
def _ctc_lambda(args):
    prediction_batch, label_batch, prediction_lengths, label_lengths = args
    # TODO check: the keras implementation takes the logarithm of the predictions (see keras.backend.ctc_batch_cost)
    return backend.ctc_batch_cost(y_true=label_batch, y_pred=prediction_batch, input_length=prediction_lengths,
                                  label_length=label_lengths)


def _compiled_wav2letter_net_with_ctc_loss(model: Sequential) -> Model:
    model = _wav2letter_net_with_ctc_loss(model)

    # TODO adjust parameters, "clipnorm seems to speeds up convergence"
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda dummy_labels, losses: losses}, optimizer=sgd)

    return model


def wav2letter_net_trained_on_batch(net: Sequential, spectrograms: List[ndarray], labels: List[str],
                                    tensor_board_log_directory: Path) -> Model:
    compiled_net_with_loss = _compiled_wav2letter_net_with_ctc_loss(net)

    training_input_dictionary = _training_input_dictionary(spectrograms, labels)

    input_batch = training_input_dictionary[InputNames.input_batch]

    def get_prediction_batch(input_batch: ndarray = input_batch) -> ndarray:
        return backend.function(net.inputs, net.outputs)([input_batch])[0]

    def print_decoded():
        prediction_batch = get_prediction_batch()
        print(decode_prediction_batch(prediction_batch))

    print_decoded()

    compiled_net_with_loss.fit(
        x=training_input_dictionary, y=_dummy_labels(batch_size=len(spectrograms)), batch_size=10, nb_epoch=100,
        callbacks=[keras.callbacks.TensorBoard(log_dir=str(tensor_board_log_directory), write_images=True)])

    print_decoded()

    return net


class InputNames:
    input_batch = "input_batch"
    label_batch = "label_batch"
    prediction_lengths = "prediction_lenghts"
    label_lengths = "label_lenghts"


def decode_prediction_batch(prediction_batch: ndarray) -> List[str]:
    # TODO use beam search with a language model instead of best path.
    return [decode_graphemes(list(argmax(prediction_batch[i], 1))) for i in range(prediction_batch.shape[0])]


def decode_graphemes(graphemes: List[int]) -> str:
    grouped_graphemes = [k for k, g in groupby(graphemes)]
    return decode_grouped_graphemes(grouped_graphemes)


def decode_grouped_graphemes(grouped_graphemes: List[int]) -> str:
    return "".join([decode_grapheme(grapheme) for grapheme in grouped_graphemes])


allowed_characters = list(string.ascii_uppercase + " '")
allowed_character_count = len(allowed_characters)
graphemes_by_character = dict((char, index) for index, char in enumerate(allowed_characters))

ctc_grapheme_set_size = len(allowed_characters) + 1
ctc_blank = ctc_grapheme_set_size - 1  # ctc blank must be last (see Tensorflow's ctcloss documentation)


def decode_grapheme(grapheme: int) -> str:
    if grapheme in range(allowed_character_count):
        return allowed_characters[grapheme]
    elif grapheme == ctc_blank:
        return ""
    else:
        raise ValueError("Unexpected grapheme: '{}'".format(grapheme))


def encode_char(label_char: chr) -> int:
    try:
        return graphemes_by_character[label_char]
    except:
        raise ValueError("Unexpected char: '{}'".format(label_char))


def _training_input_dictionary(spectrograms: List[ndarray], labels: List[str]) -> dict:
    """
    :param spectrograms: In shape (time, channels)
    :param labels:
    :return:
    """
    assert (len(labels) == len(spectrograms))

    batch_size = len(spectrograms)

    input_size_per_time_step = spectrograms[0].shape[1]

    input_lengths = [spectrogram.shape[0] for spectrogram in spectrograms]
    # Because of the striding of 2 in the network, prediction have half as many time steps as the input
    # TODO hardcoding this is error-prone, make it dependent on the network architecture
    prediction_lengths = [s / 2 for s in input_lengths]
    input_batch = zeros((batch_size, max(input_lengths), input_size_per_time_step))
    for index, spectrogram in enumerate(spectrograms):
        input_batch[index, :spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram

    label_lengths = [len(label) for label in labels]
    label_batch = -ones((batch_size, max(label_lengths)))
    for index, label in enumerate(labels):
        label_batch[index, :len(label)] = array(encode(label))

    print(input_batch.shape, label_batch.shape, input_lengths, label_lengths)
    return {
        InputNames.input_batch: input_batch,
        InputNames.label_batch: label_batch,
        InputNames.prediction_lengths: reshape(array(prediction_lengths), (batch_size, 1)),
        InputNames.label_lengths: reshape(array(label_lengths), (batch_size, 1))
    }


def encode(label: str) -> List[int]:
    return [encode_char(c) for c in label]


# for dummy loss function
def _dummy_labels(batch_size: int) -> Dict[str, ndarray]:
    return {'ctc': zeros([batch_size])}
