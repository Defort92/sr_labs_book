"""Численные проверки второй лабораторной работы."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from exercises_blank import (
    awgn,
    eval_frame_post_prob,
    frame_energy,
    framing,
    gaussian_pdf,
    gmm_train,
    load_vad_markup,
    norm_energy,
    reverb,
)
from main import calculate_metrics, detect_voice, generate_demo_signal


class VoiceActivityDetectorTests(unittest.TestCase):
    def test_rttm_markup(self):
        signal = np.zeros(1_000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "пример.rttm"
            path.write_text(
                "SPEAKER пример 1 0.2 0.3 <NA> <NA> <NA> <NA> <NA>\n",
                encoding="utf-8",
            )
            markup = load_vad_markup(path, signal, fs=1_000)

        self.assertEqual(markup.shape, signal.shape)
        self.assertEqual(markup[:200].sum(), 0)
        self.assertEqual(markup[200:500].sum(), 300)
        self.assertEqual(markup[500:].sum(), 0)

    def test_framing_and_energy(self):
        signal = np.ones(1_000, dtype=np.float32)
        frames = framing(signal, window=320, shift=160)
        energy = frame_energy(frames)

        self.assertEqual(frames.shape, (6, 320))
        np.testing.assert_allclose(energy[:5], 320)
        self.assertEqual(energy[-1], 200)

    def test_energy_normalization(self):
        normalized = norm_energy(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.std()), 1.0, places=6)

    def test_gmm_separates_low_energy_component(self):
        rng = np.random.default_rng(42)
        energy = np.concatenate(
            (
                rng.normal(-1.5, 0.15, 500),
                rng.normal(0.0, 0.2, 500),
                rng.normal(1.5, 0.15, 500),
            )
        )
        weights, means, sigmas = gmm_train(energy, gaussian_pdf, 20)
        posterior = eval_frame_post_prob(
            np.array([-1.5, 1.5]),
            gaussian_pdf,
            weights,
            means,
            sigmas,
        )

        self.assertTrue(np.all(np.diff(means) > 0))
        self.assertTrue(np.all(sigmas > 0))
        self.assertGreater(posterior[0], 0.9)
        self.assertLess(posterior[1], 0.1)

    def test_augmentations_keep_signal_length(self):
        signal = np.linspace(-1.0, 1.0, 2_000, dtype=np.float32)
        impulse_response = np.array([1.0], dtype=np.float32)
        reverberated = reverb(signal, impulse_response)
        noisy = awgn(signal, 0.1, rng=np.random.default_rng(42))

        np.testing.assert_allclose(reverberated, signal, atol=1e-6)
        self.assertEqual(noisy.shape, signal.shape)
        self.assertGreater(float(np.std(noisy - signal)), 0.09)

    def test_clean_demo_quality(self):
        signal, reference = generate_demo_signal()
        hypothesis = detect_voice(signal)
        metrics = calculate_metrics(reference, hypothesis)

        self.assertLess(metrics["fnr"], 0.05)
        self.assertLess(metrics["fpr"], 0.05)


if __name__ == "__main__":
    unittest.main()
