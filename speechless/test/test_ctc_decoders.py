# The following extract from the the ctc_beam_search_decoder documentation seems to be misleading:
# "The `ctc_greedy_decoder` is a special case of the
# `ctc_beam_search_decoder` with `top_paths=1` and `beam_width=1` (but
# that decoder is faster for this special case)."

# Instead, the following results were observed when decoding "AA<ctc_blank>AA":
#                                                           merge_repeated=True         merge_repeated=False
# tf.nn.ctc_beam_search_decoder(top_paths=1, beam_width=1)  "A"                         "AA"
# tf.nn.ctc_greedy_decoder()                                "AA"                        "AAAA"

# This is confusing and probably not intended behaviour.
# Issue was opened here: https://github.com/tensorflow/tensorflow/issues/9550

import numpy as np
import tensorflow as tf
from unittest import TestCase


class CtcDecodersTest(TestCase):
    def test(self):
        def decode_greedily(beam_search: bool, merge_repeated: bool):
            aa_ctc_blank_aa_logits = tf.constant(np.array([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]],
                                                           [[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32))
            sequence_length = tf.constant(np.array([5], dtype=np.int32))

            (decoded_list,), log_probabilities = \
                tf.nn.ctc_beam_search_decoder(inputs=aa_ctc_blank_aa_logits,
                                              sequence_length=sequence_length,
                                              merge_repeated=merge_repeated,
                                              beam_width=1) \
                    if beam_search else \
                    tf.nn.ctc_greedy_decoder(inputs=aa_ctc_blank_aa_logits,
                                             sequence_length=sequence_length,
                                             merge_repeated=merge_repeated)

            return list(tf.Session().run(tf.sparse_tensor_to_dense(decoded_list)[0]))

        self.assertEqual([0], decode_greedily(beam_search=True, merge_repeated=True))
        self.assertEqual([0, 0], decode_greedily(beam_search=True, merge_repeated=False))
        self.assertEqual([0, 0], decode_greedily(beam_search=False, merge_repeated=True))
        self.assertEqual([0, 0, 0, 0], decode_greedily(beam_search=False, merge_repeated=False))
