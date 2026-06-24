"""
Tests voor feature engineering (dataVerwerken/preprocess_data.py).

Controleert dat de afgeleide features correct berekend worden uit ruwe data.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataVerwerken.preprocess_data import add_features


def make_raw_df(n=100):
    """Bouw een ruwe order-book DataFrame met oplopende prijs."""
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame({
        'close': close,
        'volume': np.full(n, 1.0),
        'bid_price': close - 0.5,
        'ask_price': close + 0.5,
        'bid_volume': np.full(n, 2.0),
        'ask_volume': np.full(n, 1.0),
    })


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.df = add_features(make_raw_df())

    def test_spread_is_ask_min_bid(self):
        """spread = ask − bid, hier constant 1.0."""
        # ask - bid = (close+0.5) - (close-0.5) = 1.0
        np.testing.assert_allclose(self.df['spread'].dropna(), 1.0, atol=1e-6)

    def test_order_imbalance_bereik(self):
        """order_imbalance = (bid_vol − ask_vol)/(bid_vol + ask_vol), hier 1/3."""
        # (2 - 1) / (2 + 1) = 0.333...
        vals = self.df['order_imbalance'].dropna()
        np.testing.assert_allclose(vals, 1.0 / 3.0, atol=1e-6)

    def test_rsi_in_bereik_0_100(self):
        """RSI ligt altijd binnen het geldige bereik [0, 100]."""
        rsi = self.df['rsi_14'].dropna()
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())

    def test_rsi_hoog_bij_stijgende_prijs(self):
        """Bij monotoon stijgende prijs ligt RSI dicht bij 100 (>90)."""
        # Prijs stijgt monotoon → alleen winsten → RSI dicht bij 100.
        rsi = self.df['rsi_14'].dropna()
        self.assertGreater(rsi.iloc[-1], 90.0)

    def test_returns_positief_bij_stijgende_prijs(self):
        """Bij stijgende prijs is de 5s-return positief."""
        self.assertTrue((self.df['return_5s'].dropna() > 0).all())

    def test_momentum_positief_bij_stijgende_prijs(self):
        """Bij stijgende prijs is momentum_10 positief."""
        self.assertTrue((self.df['momentum_10'].dropna() > 0).all())


if __name__ == '__main__':
    unittest.main()
