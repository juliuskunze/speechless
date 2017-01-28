from unittest import TestCase

from net import decode_grouped_graphemes, encode, ctc_blank, decode_graphemes


class NetTest(TestCase):
    def test_encode(self):
        label = "SHE WASN'T THREE ABCXYZ"
        self.assertEqual(label, decode_grouped_graphemes(encode(label)))

    def test_decode(self):
        graphemes = encode("SSSSHHHHEEEEE      WASN'T THRE") + [ctc_blank] + encode("EEEEEE")
        self.assertEqual("SHE WASN'T THREE", decode_graphemes(graphemes))
