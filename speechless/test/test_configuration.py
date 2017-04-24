from pathlib import Path

from unittest import TestCase

from speechless.configuration import LoggedRun
from speechless.tools import log


class LoggedRunTest(TestCase):
    def test(self):
        l1 = LoggedRun(lambda: log("1"), "test1", Path())
        l1()

        self.assertEqual("1\n", l1.result_file.read_text())

        l2 = LoggedRun(lambda: log("2"), "test2", Path())
        l2()

        self.assertEqual("1\n", l1.result_file.read_text())
        self.assertEqual("2\n", l2.result_file.read_text())

        l1.result_file.unlink()
        l2.result_file.unlink()
