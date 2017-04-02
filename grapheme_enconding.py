import string
from itertools import groupby

from numpy import argmax, ones, ndarray, array
from typing import List

frequent_characters_in_english = list(string.ascii_lowercase + " '")
frequent_characters_in_german = frequent_characters_in_english + list("äöüß-")

class CtcGraphemeEncoding:
    def __init__(self, allowed_characters: List[chr] = frequent_characters_in_english):
        self.graphemes_by_character = dict((char, index) for index, char in enumerate(allowed_characters))
        self.allowed_characters = allowed_characters
        self.grapheme_set_size = len(allowed_characters) + 1
        self.allowed_character_count = len(allowed_characters)
        # ctc blank must be last (see Tensorflow's ctcloss documentation):
        self.ctc_blank = self.grapheme_set_size - 1

    def decode_prediction_batch(self, prediction_batch: ndarray, prediction_lengths: List[int]) -> List[str]:
        """
        :param prediction_batch: In shape (example, time, grapheme).
        :param prediction_lengths:
        :return:
        """
        # TODO use beam search with a language model instead of best path.
        return [self.decode_graphemes(list(argmax(prediction_batch[i], 1))[:prediction_lengths[i]]) for i in
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

    def encode_label_batch(self, labels: List[str]):
        batch_size = len(labels)
        label_lengths = [len(label) for label in labels]
        label_batch = -ones((batch_size, max(label_lengths)))
        for index, label in enumerate(labels):
            label_batch[index, :len(label)] = array(self.encode(label))

        return label_batch
