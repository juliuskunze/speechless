from unittest import TestCase

from speechless.tools import paginate


class ToolsTest(TestCase):
    def test_paginate(self):
        a = paginate([1, 2, 3], 2)
        self.assertEqual(list(a), [[1, 2], [3, ]])
