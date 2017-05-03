from abc import abstractmethod
from itertools import groupby

from numpy import argmax, ones, ndarray, array
from typing import List


class GraphemeEncodingBase:
    def __init__(self, allowed_characters: List[chr], special_grapheme_count: int):
        self.allowed_characters = allowed_characters
        self.allowed_character_count = len(allowed_characters)
        self.grapheme_set_size = self.allowed_character_count + special_grapheme_count
        self.graphemes_by_character = dict((char, index) for index, char in enumerate(allowed_characters))

    def encode_character(self, label_char: chr) -> int:
        try:
            return self.graphemes_by_character[label_char]
        except:
            raise ValueError("Unexpected char: '{}'".format(label_char))

    @abstractmethod
    def encode(self, label: str) -> List[int]:
        pass

    def encode_label_batch(self, labels: List[str]):
        batch_size = len(labels)
        label_lengths = [len(label) for label in labels]
        label_batch = -ones((batch_size, max(label_lengths)), dtype='int32')
        for index, label in enumerate(labels):
            label_batch[index, :len(label)] = array(self.encode(label))

        return label_batch

    def decode_graphemes(self, graphemes: List[int], merge_repeated: bool = True) -> str:
        if merge_repeated:
            graphemes = [k for k, g in groupby(graphemes)]
        return "".join([self.decode_grapheme(grapheme,
                                             previous_grapheme=graphemes[index - 1] if index > 0 else None)
                        for index, grapheme in enumerate(graphemes)])

    def decode_prediction_batch(self, prediction_batch: ndarray, prediction_lengths: List[int]) -> List[str]:
        """
        :param prediction_batch: In shape (example.py, time, grapheme).
        :param prediction_lengths:
        :return:
        """
        return self.decode_grapheme_batch(argmax(prediction_batch, 2), prediction_lengths)

    def decode_grapheme_batch(self, grapheme_batch: ndarray, prediction_lengths: List[int],
                              merge_repeated: bool = True) -> List[str]:
        """
        :param grapheme_batch: In shape (example.py, time).
        :param prediction_lengths:
        :return:
        """
        return [self.decode_graphemes(list(grapheme_batch[i])[:prediction_lengths[i]], merge_repeated=merge_repeated)
                for i in range(grapheme_batch.shape[0])]

    @abstractmethod
    def decode_grapheme(self, grapheme: int, previous_grapheme: int) -> str:
        pass


class AsgGraphemeEncoding(GraphemeEncodingBase):
    def __init__(self, allowed_characters: List[chr]):
        super().__init__(allowed_characters, special_grapheme_count=2)

        self.asg_twice = self.grapheme_set_size - 2
        self.asg_thrice = self.grapheme_set_size - 1

    def encode(self, label: str) -> List[int]:
        naive_encoded = [self.encode_character(c) for c in label]

        def repetition_count_after(index: int) -> int:
            original_grapheme = naive_encoded[index]
            result = 1
            while True:
                index += 1

                if index == len(naive_encoded):
                    return result

                grapheme = naive_encoded[index]

                if grapheme != original_grapheme:
                    return result

                result += 1

        encoded = []
        index = 0
        while index < len(naive_encoded):
            repetition_count = repetition_count_after(index)
            encoded.append(naive_encoded[index])
            index += repetition_count
            if repetition_count == 1:
                continue
            if repetition_count == 2:
                encoded.append(self.asg_twice)
            elif repetition_count == 3:
                encoded.append(self.asg_thrice)
            else:
                raise ValueError("{}-fold repetition found, ASG only supports up to 3-fold.".format(repetition_count))

        return encoded

    def decode_grapheme(self, grapheme: int, previous_grapheme: int) -> str:
        if grapheme in range(self.allowed_character_count):
            return self.allowed_characters[grapheme]
        elif grapheme == self.asg_twice:
            return self.allowed_characters[previous_grapheme]
        elif grapheme == self.asg_thrice:
            if previous_grapheme is None or previous_grapheme not in range(self.allowed_character_count):
                return ""

            return "".join([self.allowed_characters[previous_grapheme]] * 2)
        else:
            raise ValueError("Unexpected grapheme: '{}'".format(grapheme))


class CtcGraphemeEncoding(GraphemeEncodingBase):
    def __init__(self, allowed_characters: List[chr]):
        super().__init__(allowed_characters, special_grapheme_count=1)

        # ctc blank must be last (see Tensorflow's ctcloss documentation):
        self.ctc_blank = self.grapheme_set_size - 1

    def encode(self, label: str) -> List[int]:
        return [self.encode_character(c) for c in label]

    def decode_grapheme(self, grapheme: int, previous_grapheme: int) -> str:
        if grapheme in range(self.allowed_character_count):
            return self.allowed_characters[grapheme]
        elif grapheme == self.ctc_blank:
            return ""
        else:
            raise ValueError("Unexpected grapheme: '{}'".format(grapheme))
