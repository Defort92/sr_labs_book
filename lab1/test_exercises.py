"""Небольшие численные проверки для первой лабораторной работы."""

import unittest

import numpy as np

from exercises_blank import (
    compute_fbank_filters,
    compute_fbanks_features,
    compute_mfcc,
    framing,
    mvn_floating,
    power_spectrum,
    preemphasis,
    split_meta_line,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_metadata_parser(self):
        self.assertEqual(
            split_meta_line("id00001 m ../data/example.wav\n"),
            ("id00001", "m", "../data/example.wav"),
        )

    def test_preemphasis(self):
        signal = np.array([1.0, 2.0, 4.0])
        np.testing.assert_allclose(preemphasis(signal), [1.0, 1.03, 2.06])

    def test_pipeline_shapes_and_finite_values(self):
        sample_rate = 16_000
        time = np.arange(sample_rate) / sample_rate
        signal = np.sin(2 * np.pi * 440 * time)

        frames = framing(preemphasis(signal), sample_rate=sample_rate)
        spectrum = power_spectrum(frames, NFFT=512)
        filters = compute_fbank_filters(40, sample_rate, 512)
        filter_banks = compute_fbanks_features(spectrum, filters)
        mfcc = compute_mfcc(filter_banks, 20)

        self.assertEqual(frames.shape, (99, 400))
        self.assertEqual(spectrum.shape, (99, 257))
        self.assertEqual(filters.shape, (40, 257))
        self.assertEqual(filter_banks.shape, (99, 40))
        self.assertEqual(mfcc.shape, (99, 20))
        self.assertTrue(np.isfinite(filter_banks).all())
        self.assertTrue(np.isfinite(mfcc).all())

    def test_mvn_normalizes_global_window(self):
        features = np.arange(30, dtype=float).reshape(10, 3)
        normalized = mvn_floating(features, 20, 20)
        np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(normalized.std(axis=0), 1, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
