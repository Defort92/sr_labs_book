"""Автоматические проверки четвёртой лабораторной работы."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

LAB4_DIR = Path(__file__).resolve().parent
if str(LAB4_DIR) not in sys.path:
    sys.path.insert(0, str(LAB4_DIR))

from constants import (
    ENCODER_ATTENTIVE_STATISTICS_POOLING,
    ENCODER_SELF_ATTENTIVE_POOLING,
)
from exercises_blank import (
    bayes_test,
    equal_error_rate,
    gaussian_pdf,
    llr,
    map_test,
    minmax_test,
    neyman_pearson_test,
    tar_imp_hists,
)
from ResNetBlocks import SEBasicBlock
from ResNetSE34V2 import ResNetSE


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class DecisionCriteriaTests(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([0.82, 0.73, 0.66, 0.20, 0.10, -0.05])
        self.labels = np.array([1, 1, 1, 0, 0, 0])
        self.target_scores, self.impostor_scores = tar_imp_hists(
            self.scores,
            self.labels,
        )
        (
            self.sorted_labels,
            self.sorted_scores,
            _,
            _,
            self.log_likelihood_ratio,
        ) = llr(
            self.scores,
            self.labels,
            self.target_scores,
            self.impostor_scores,
            gaussian_pdf,
        )

    def test_scores_are_split_and_sorted(self):
        np.testing.assert_array_equal(
            self.target_scores,
            np.array([0.82, 0.73, 0.66]),
        )
        np.testing.assert_array_equal(
            self.impostor_scores,
            np.array([0.20, 0.10, -0.05]),
        )
        self.assertTrue(np.all(np.diff(self.sorted_scores) >= 0.0))
        self.assertTrue(np.isfinite(self.log_likelihood_ratio).all())

    def test_map_and_neyman_pearson_find_separating_threshold(self):
        _, fnr, fpr, risk = map_test(
            self.sorted_labels,
            self.log_likelihood_ratio,
            self.target_scores,
            self.impostor_scores,
            0.5,
        )
        self.assertEqual(fnr, 0.0)
        self.assertEqual(fpr, 0.0)
        self.assertEqual(risk, 0.0)

        _, constrained_fpr = neyman_pearson_test(
            self.sorted_labels,
            self.log_likelihood_ratio,
            self.target_scores,
            self.impostor_scores,
            0.25,
        )
        self.assertEqual(constrained_fpr, 0.0)

    def test_bayes_and_minimax_return_valid_values(self):
        threshold, fnr, fpr, risk = bayes_test(
            self.sorted_labels,
            self.log_likelihood_ratio,
            self.target_scores,
            self.impostor_scores,
            0.5,
            0.0,
            1.0,
            1.0,
            0.0,
        )
        self.assertAlmostEqual(threshold, 0.0)
        self.assertGreaterEqual(fnr, 0.0)
        self.assertGreaterEqual(fpr, 0.0)
        self.assertGreaterEqual(risk, 0.0)

        result = minmax_test(
            self.sorted_labels,
            self.log_likelihood_ratio,
            self.target_scores,
            self.impostor_scores,
            np.linspace(0.0, 1.0, 51),
            0.0,
            1.0,
            1.0,
            0.0,
        )
        self.assertEqual(len(result), 5)
        self.assertTrue(0.0 <= result[4] <= 1.0)

    def test_eer_is_zero_for_non_overlapping_scores(self):
        eer, _ = equal_error_rate(self.labels.astype(bool), self.scores)
        self.assertEqual(eer, 0.0)

    def test_resnet_se_returns_fixed_size_embedding(self):
        audio = torch.randn(2, 4_000)
        for encoder_type in (
            ENCODER_SELF_ATTENTIVE_POOLING,
            ENCODER_ATTENTIVE_STATISTICS_POOLING,
        ):
            model = ResNetSE(
                SEBasicBlock,
                layers=(1, 1, 1, 1),
                num_filters=(8, 16, 32, 64),
                nOut=10,
                encoder_type=encoder_type,
                n_mels=16,
            )
            model.eval()
            with torch.no_grad():
                embeddings = model(audio)
            self.assertEqual(embeddings.shape, (2, 10))
            self.assertTrue(torch.isfinite(embeddings).all())


if __name__ == "__main__":
    unittest.main()
