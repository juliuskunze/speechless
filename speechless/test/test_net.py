from unittest import TestCase

from speechless.net import ExpectationsVsPredictionsInGroupedBatches, ExpectationsVsPredictionsInBatches, \
    ExpectationsVsPredictions, ExpectationVsPrediction


class NetTest(TestCase):
    def test_sanity_expectation_vs_prediction(self):
        a = ExpectationVsPrediction(expected="A", predicted="A", loss=0.0)
        b = ExpectationVsPrediction(expected="B", predicted="A", loss=2.0)
        results_batches = [ExpectationsVsPredictions([a, b]),
                           ExpectationsVsPredictions([])]

        results_by_name = ExpectationsVsPredictionsInBatches(result_batches=results_batches)
        e = ExpectationsVsPredictionsInGroupedBatches(
            results_by_group_name=dict([
                ("corpus1", results_by_name),
                ("corpus2", results_by_name),
                ("empty", ExpectationsVsPredictionsInBatches([]))]))

        print(str(e))
