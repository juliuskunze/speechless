from typing import List, Dict

from keras import backend
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution1D, Lambda
from keras.models import Sequential
from keras.optimizers import SGD
from numpy import *


def wav2letter_net(
        input_size_per_timestep: int,
        output_charset_size: int,
        use_raw_wave_input: bool = False,
        activation: str = "relu",
        output_activation: str = None
) -> Sequential:
    """As described in https://arxiv.org/pdf/1609.03193v2.pdf"""
    main_filter_count = 250

    def input_convolutions() -> List[Convolution1D]:
        raw_wave_convolution_if_needed = [
            Convolution1D(nb_filter=main_filter_count, filter_length=250, subsample_length=160, activation=activation,
                          name="wave_conv", input_dim=input_size_per_timestep)] if use_raw_wave_input else []

        return raw_wave_convolution_if_needed + [
            Convolution1D(nb_filter=main_filter_count, filter_length=48, subsample_length=2, activation=activation,
                          name="striding_conv", input_dim=None if use_raw_wave_input else input_size_per_timestep)]

    def inner_convolutions() -> List[Convolution1D]:
        return [Convolution1D(nb_filter=main_filter_count, filter_length=7, activation=activation,
                              name="inner_conv_{}".format(i)) for i in range(1, 8)]

    def output_convolutions() -> List[Convolution1D]:
        out_filter_count = 2000
        return [
            Convolution1D(nb_filter=out_filter_count, filter_length=32, activation=activation, name="big_conv_1"),
            Convolution1D(nb_filter=out_filter_count, filter_length=1, activation=activation, name="big_conv_2"),
            Convolution1D(nb_filter=output_charset_size, filter_length=1, activation=output_activation,
                          name="output_conv")
        ]

    return Sequential(
        input_convolutions() +
        inner_convolutions() +
        output_convolutions())


def _wav2letter_with_ctc_cost(model: Sequential) -> Model:
    input_batch = Input(name=_InputNames.input_batch, shape=model.input_shape)
    # TODO shape=[None], this may fail, replace by "absolute max string length"?
    labels = Input(name=_InputNames.labels, shape=[None])
    input_lengths = Input(name=_InputNames.input_lengths, shape=[1], dtype='int64')
    label_lengths = Input(name=_InputNames.label_lengths, shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    y_pred = model(input_batch)
    loss_out = Lambda(_ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_lengths, label_lengths])
    return Model(input=[input_batch, labels, input_lengths, label_lengths], output=[loss_out])


def _wav2letter_compiled(model: Sequential) -> Model:
    model = _wav2letter_with_ctc_cost(model)

    # TODO clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    return model.compile(
        loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)


def wav2letter_trained_on_batch(net: Sequential, spectrograms: List[ndarray], labels: List[str]) -> Model:
    net = _wav2letter_compiled(net)

    net.train_on_batch(x=_input_dictionary(spectrograms, labels), y=_dummy_outputs(batch_size=len(spectrograms)))

    # TODO callbacks = [keras.callbacks.TensorBoard(log_dir=str(Path(base_directory, 'logs')), write_graph=True, write_images=False)]
    return net


class _InputNames:
    input_batch = "input_batch"
    labels = "labels"
    input_lengths = "input_lenghts"
    label_lengths = "label_lenghts"


def _input_dictionary(spectrograms: List[ndarray], labels: List[str]):
    """
    :param spectrograms: In shape (time, channels)
    :param labels:
    :return:
    """
    assert (len(labels) == len(spectrograms))

    batch_size = len(spectrograms)

    input_lengths = [x.shape[0] for x in spectrograms]
    target_shape = (len(spectrograms), (max(input_lengths)), spectrograms[0].shape[1])
    # shape: (batch, time, channels)
    input_batch = zeros(target_shape)
    for index, spectrogram in enumerate(spectrograms):
        input_batch[index, :spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram
    return {
        _InputNames.input_batch: input_batch,
        _InputNames.labels: labels,
        _InputNames.input_lengths: reshape(array(input_lengths), (batch_size, 1)),
        _InputNames.label_lengths: reshape(array([len(label) for label in labels]), (batch_size, 1))
    }


# the actual loss calc occurs here despite it not being
# an internal Keras loss function
def _ctc_lambda_func(args: (int, int, ndarray, ndarray)) -> ndarray:
    y_pred, labels, input_lengths, label_lengths = args
    return backend.ctc_batch_cost(y_true=labels, y_pred=y_pred, input_length=input_lengths, label_length=label_lengths)


def _dummy_outputs(batch_size: int) -> Dict[str, ndarray]:
    return {'ctc': zeros([batch_size])}  # dummy data for dummy loss function
