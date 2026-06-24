"""
Non-functionele tests (performance & efficiëntie).

Deze tests controleren NIET of de uitkomst functioneel klopt, maar of de code
aan kwaliteitseisen voldoet: snelheid (throughput/latency) en geheugen-
efficiëntie (zero-copy sliding window, efficiënte data-structuur).

Testtechniek: drempelwaarde-tests met ruime marges, zodat ze niet 'flaky'
worden op trage CI-machines maar wel regressies in efficiëntie vangen.
"""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.envs.trading_env import CryptoTradingEnv
from src.models.deeplob import DeepLOB


def make_env(n=5000, window=100, n_features=15):
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((n, n_features)).astype(np.float32)
    prices = np.full(n, 100.0, dtype=np.float64)
    return CryptoTradingEnv(
        raw_features=raw, prices=prices, window_size=window,
        transaction_fee=0.0, flat_fee=1.0, discrete_actions=True,
    ), raw


class TestPerformance(unittest.TestCase):
    """Snelheid: de env en het model moeten ruim snel genoeg zijn."""

    def test_env_throughput(self):
        """De env verwerkt minstens 2000 steps/seconde (ruime ondergrens)."""
        env, _ = make_env(n=5000, window=100)
        env.reset()
        n_steps = 0
        start = time.time()
        for _ in range(4000):
            _, _, term, trunc, _ = env.step(0)
            n_steps += 1
            if term or trunc:
                env.reset()
        elapsed = time.time() - start
        throughput = n_steps / elapsed
        self.assertGreater(throughput, 2000.0,
                           f"Te traag: {throughput:.0f} steps/s")

    def test_deeplob_inference_latency(self):
        """DeepLOB verwerkt een batch van 32 in minder dan 2 seconden (CPU)."""
        model = DeepLOB(input_dim=15, output_dim=64).eval()
        x = torch.randn(32, 100, 15)
        start = time.time()
        with torch.no_grad():
            model(x)
        self.assertLess(time.time() - start, 2.0)


class TestGeheugenEfficientie(unittest.TestCase):
    """Geheugen: de sliding window mag de data niet 100x dupliceren."""

    def test_sliding_window_is_zero_copy(self):
        """Een window-slice deelt geheugen met de ruwe array (geen kopie)."""
        _, raw = make_env(n=500, window=100)
        window = raw[0:100]
        self.assertTrue(np.shares_memory(window, raw))

    def test_env_bewaart_ruwe_array_zonder_kopie(self):
        """De env bewaart de ruwe (N, features) array, niet een uitgerolde kopie."""
        env, raw = make_env(n=500, window=100)
        self.assertIs(env._raw_features, raw)

    def test_streaming_veel_zuiniger_dan_materialiseren(self):
        """Streaming-opslag is ~window_size keer kleiner dan uitgerolde sequences."""
        n, window, feat = 5000, 100, 15
        env, raw = make_env(n=n, window=window)
        raw_bytes = env._raw_features.nbytes
        # Een volledig gematerialiseerde (N-window, window, feat) array zou kosten:
        materialised_bytes = (n - window) * window * feat * 4  # float32
        ratio = materialised_bytes / raw_bytes
        self.assertGreater(ratio, window / 2,
                           f"Streaming bespaart te weinig geheugen (factor {ratio:.0f})")


if __name__ == "__main__":
    unittest.main()
