"""
Acceptatietests (end-to-end, vanuit klant-/gebruikersperspectief).

Waar unit tests losse onderdelen controleren, controleren deze tests of een
volledige gebruikersscenario van begin tot eind werkt — geformuleerd als
'gegeven / wanneer / dan' (Given-When-Then) user stories.

Deze tests gebruiken alleen synthetische data, zodat ze overal draaien zonder
de echte coreData-dataset.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.trading_env import CryptoTradingEnv, FlatCryptoTradingEnv


def make_data(n=400, n_features=15, prijs=None):
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n, n_features)).astype(np.float32)
    if prijs is None:
        prijs = np.full(n, 100.0, dtype=np.float64)
    return raw, prijs


class TestAcceptatie(unittest.TestCase):
    """End-to-end scenario's zoals een gebruiker de env zou inzetten."""

    def test_volledige_episode_loopt_zonder_fouten(self):
        """Gegeven een env, wanneer een agent een hele episode speelt, dan eindigt die netjes."""
        raw, prijs = make_data(n=300, n_features=15)
        env = CryptoTradingEnv(raw_features=raw, prices=prijs, window_size=100,
                               discrete_actions=True)
        env.reset(seed=0)
        klaar = False
        stappen = 0
        for _ in range(1000):
            actie = env.action_space.sample()
            _, reward, term, trunc, info = env.step(actie)
            self.assertTrue(np.isfinite(reward))  # reward blijft eindig
            stappen += 1
            if term or trunc:
                klaar = True
                break
        self.assertTrue(klaar, "Episode moet binnen de data eindigen")
        self.assertGreater(stappen, 0)

    def test_eindrapport_metrics_zijn_geldig(self):
        """Na een episode levert de env bruikbare, geldige metrics op voor rapportage."""
        raw, prijs = make_data(n=300)
        env = CryptoTradingEnv(raw_features=raw, prices=prijs, window_size=100,
                               discrete_actions=True)
        env.reset(seed=1)
        info = {}
        for _ in range(1000):
            _, _, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                break
        self.assertTrue(np.isfinite(info['portfolio_value']))
        self.assertGreaterEqual(info['portfolio_value'], 0.0)
        self.assertGreaterEqual(info['total_trades'], 0)
        self.assertTrue(0.0 <= info['win_rate'] <= 1.0)
        self.assertTrue(np.isfinite(info['total_return']))

    def test_buy_and_hold_bij_stijgende_prijs_geeft_winst(self):
        """Gegeven een stijgende prijs, wanneer je koopt en vasthoudt, dan maak je winst."""
        n = 300
        prijs = np.linspace(100.0, 200.0, n)  # prijs verdubbelt
        raw, _ = make_data(n=n)
        env = CryptoTradingEnv(raw_features=raw, prices=prijs, window_size=100,
                               discrete_actions=True, flat_fee=1.0)
        env.reset()
        env.step(1)  # Buy aan het begin
        info = {}
        for _ in range(1000):
            _, _, term, trunc, info = env.step(0)  # Hold tot het einde
            if term or trunc:
                break
        self.assertGreater(info['total_return'], 0.0,
                           "Buy-and-hold bij stijgende prijs moet winst opleveren")

    def test_flat_env_werkt_end_to_end_voor_mlp(self):
        """De flat-variant (voor MLP) levert een platte vector en draait een episode af."""
        raw, prijs = make_data(n=250)
        env = FlatCryptoTradingEnv(raw_features=raw, prices=prijs, window_size=100,
                                   discrete_actions=True)
        obs, _ = env.reset()
        self.assertEqual(obs.ndim, 1)  # platte vector
        self.assertEqual(obs.shape[0], 100 * 15 + 4)
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            self.assertEqual(obs.ndim, 1)
            if term or trunc:
                break


if __name__ == "__main__":
    unittest.main()
