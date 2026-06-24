"""
Tests voor CryptoTradingEnv.

Controleert de kernlogica van de trading environment: reset, step, kopen,
verkopen, portfoliowaarde en observatievormen.
"""

import os
import sys
import unittest

import numpy as np

# Maak de projectroot importeerbaar (zodat `src...` werkt).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.trading_env import CryptoTradingEnv


def make_env(n=200, window=50, n_features=15, initial_balance=100_000.0, flat_fee=1.0):
    """Bouw een kleine env met synthetische data (constante prijs $100)."""
    rng = np.random.default_rng(0)
    raw_features = rng.standard_normal((n, n_features)).astype(np.float32)
    prices = np.full(n, 100.0, dtype=np.float64)
    return CryptoTradingEnv(
        raw_features=raw_features,
        prices=prices,
        window_size=window,
        initial_balance=initial_balance,
        flat_fee=flat_fee,
        transaction_fee=0.0,
        discrete_actions=True,
    )


class TestReset(unittest.TestCase):

    def test_reset_geeft_startkapitaal(self):
        """Na reset staat de balans op het startkapitaal en is er geen BTC."""
        env = make_env()
        obs, info = env.reset()
        self.assertEqual(env.balance, 100_000.0)
        self.assertEqual(env.btc_held, 0.0)

    def test_observatie_vorm(self):
        """De observatie heeft de juiste vorm: features (50,15) en portfolio (4,)."""
        env = make_env(window=50, n_features=15)
        obs, _ = env.reset()
        self.assertEqual(obs['features'].shape, (50, 15))
        self.assertEqual(obs['portfolio'].shape, (4,))

    def test_observatie_dtype_float32(self):
        """Beide observatie-arrays zijn van type float32 (klaar voor PyTorch)."""
        env = make_env()
        obs, _ = env.reset()
        self.assertEqual(obs['features'].dtype, np.float32)
        self.assertEqual(obs['portfolio'].dtype, np.float32)


class TestStep(unittest.TestCase):

    def test_hold_verandert_balans_niet(self):
        """Actie Hold (0) doet geen transactie: balans en BTC blijven gelijk."""
        env = make_env()
        env.reset()
        balance_voor = env.balance
        obs, reward, terminated, truncated, info = env.step(0)  # Hold
        self.assertEqual(env.balance, balance_voor)
        self.assertEqual(env.btc_held, 0.0)

    def test_step_geeft_vijf_waarden(self):
        """step() geeft de Gymnasium-tuple terug: (obs, reward, terminated, truncated, info)."""
        env = make_env()
        env.reset()
        result = env.step(0)
        self.assertEqual(len(result), 5)

    def test_buy_zet_cash_om_in_btc(self):
        """Actie Buy (1) zet cash om in BTC: btc_held stijgt, balans daalt."""
        env = make_env()
        env.reset()
        env.step(1)  # Buy
        self.assertGreater(env.btc_held, 0.0)
        self.assertLess(env.balance, 100_000.0)

    def test_buy_dan_sell_sluit_positie(self):
        """Buy gevolgd door Sell (2) sluit de positie: btc_held terug naar 0."""
        env = make_env()
        env.reset()
        env.step(1)  # Buy
        self.assertGreater(env.btc_held, 0.0)
        env.step(2)  # Sell
        self.assertEqual(env.btc_held, 0.0)

    def test_episode_eindigt_aan_einde_data(self):
        """De episode eindigt (truncated=True) zodra de data op is."""
        env = make_env(n=120, window=50)  # 70 stappen mogelijk
        env.reset()
        truncated = False
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        self.assertTrue(truncated)


class TestPortfolio(unittest.TestCase):

    def test_portfoliowaarde_bij_start(self):
        """De portfoliowaarde bij start is gelijk aan het startkapitaal."""
        env = make_env(initial_balance=50_000.0)
        env.reset()
        self.assertAlmostEqual(env._calculate_portfolio_value(), 50_000.0, places=2)

    def test_kosten_verlagen_portfolio_bij_constante_prijs(self):
        """Bij constante prijs verlaagt de transactiekost (flat_fee) de portfoliowaarde."""
        # Bij constante prijs kost een buy alleen de flat_fee → waarde daalt licht.
        env = make_env(flat_fee=1.0)
        env.reset()
        waarde_voor = env._calculate_portfolio_value()
        env.step(1)  # Buy
        waarde_na = env._calculate_portfolio_value()
        self.assertLess(waarde_na, waarde_voor)
        self.assertAlmostEqual(waarde_voor - waarde_na, 1.0, delta=0.5)

    def test_info_bevat_kernmetrics(self):
        """De info-dict bevat de kernmetrics (portfolio_value, return, trades, drawdown)."""
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(0)
        for key in ['portfolio_value', 'total_return', 'total_trades', 'max_drawdown']:
            self.assertIn(key, info)


if __name__ == '__main__':
    unittest.main()
