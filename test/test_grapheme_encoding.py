from numpy import *
from typing import List
from unittest import TestCase

from grapheme_enconding import CtcGraphemeEncoding, AsgGraphemeEncoding


class CtcGraphemeEncodingTests(TestCase):
    def test_encode(self):
        g = CtcGraphemeEncoding()
        label = "she wasn't three abcxyz"
        self.assertEqual(label, g.decode_grouped_graphemes(g.encode(label)))

    def test_decode(self):
        g = CtcGraphemeEncoding()
        graphemes = g.encode("sssshhhheeeee      wasn't thre") + [g.ctc_blank] + g.encode("eeeeee")
        self.assertEqual("she wasn't three", g.decode_graphemes(graphemes))

    def test_encode_batch(self):
        g = CtcGraphemeEncoding()
        predictions = zeros((2, 3, g.grapheme_set_size))
        predictions[0, 0, g.encode_character("a")] = 1
        predictions[0, 1, g.encode_character("b")] = 1
        predictions[0, 2, g.encode_character("c")] = 1

        predictions[1, 0, g.encode_character("a")] = 1
        predictions[1, 1, g.encode_character("b")] = 1
        predictions[1, 2, g.encode_character("c")] = 1

        self.assertEqual(["abc", "ab"], g.decode_prediction_batch(predictions, prediction_lengths=[3, 2]))


class AsgGraphemeEncodingTests(TestCase):
    def test_encode_repetitions(self):
        g = AsgGraphemeEncoding()
        self.assertEqual([g.encode_character("e"), g.asg_twice], g.encode("ee"))
        self.assertEqual([g.encode_character("e"), g.asg_thrice], g.encode("eee"))
        with self.assertRaises(ValueError):
            g.encode("eeee")

    def test_decode(self):
        g = AsgGraphemeEncoding()

        def encode_char_by_char(label: str) -> List[int]:
            return [g.encode_character(c) for c in label]

        graphemes = encode_char_by_char("sssshhhheeeee      wasn't thre") + [g.asg_twice, g.asg_twice, g.asg_twice] + \
                    encode_char_by_char("    aaaaaaa") + [g.asg_thrice]
        self.assertEqual("she wasn't three aaa", g.decode_graphemes(graphemes))
