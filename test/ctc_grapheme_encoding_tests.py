from unittest import TestCase

from numpy import *

from grapheme_enconding import CtcGraphemeEncoding


class CtcGraphemeEncodingTests(TestCase):
    def test_encode(self):
        g = CtcGraphemeEncoding()
        label = "SHE WASN'T THREE ABCXYZ"
        self.assertEqual(label, g.decode_grouped_graphemes(g.encode(label)))

    def test_decode(self):
        g = CtcGraphemeEncoding()
        graphemes = g.encode("SSSSHHHHEEEEE      WASN'T THRE") + [g.ctc_blank] + g.encode("EEEEEE")
        self.assertEqual("SHE WASN'T THREE", g.decode_graphemes(graphemes))

    def test_encode_batch(self):
        g = CtcGraphemeEncoding()
        predictions = zeros((2, 3, g.grapheme_set_size))
        predictions[0, 0, g.encode_char("A")] = 1
        predictions[0, 1, g.encode_char("B")] = 1
        predictions[0, 2, g.encode_char("C")] = 1

        predictions[1, 0, g.encode_char("A")] = 1
        predictions[1, 1, g.encode_char("B")] = 1
        predictions[1, 2, g.encode_char("C")] = 1

        self.assertEqual(["ABC", "AB"], g.decode_prediction_batch(predictions, prediction_lengths=[3, 2]))
