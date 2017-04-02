from numpy import *
from unittest import TestCase

from grapheme_enconding import CtcGraphemeEncoding


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
        predictions[0, 0, g.encode_char("a")] = 1
        predictions[0, 1, g.encode_char("b")] = 1
        predictions[0, 2, g.encode_char("c")] = 1

        predictions[1, 0, g.encode_char("a")] = 1
        predictions[1, 1, g.encode_char("b")] = 1
        predictions[1, 2, g.encode_char("c")] = 1

        self.assertEqual(["abc", "ab"], g.decode_prediction_batch(predictions, prediction_lengths=[3, 2]))
