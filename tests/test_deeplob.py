"""
Tests voor het DeepLOB-model.

Controleert dat de forward pass werkt en de juiste outputvorm geeft.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models.deeplob import DeepLOB


class TestDeepLOB(unittest.TestCase):

    def setUp(self):
        self.batch = 4
        self.seq_len = 100
        self.n_features = 15
        self.output_dim = 64
        self.model = DeepLOB(
            input_dim=self.n_features,
            output_dim=self.output_dim,
        )
        self.model.eval()
        self.x = torch.randn(self.batch, self.seq_len, self.n_features)

    def test_output_vorm(self):
        """De forward pass geeft de juiste outputvorm (batch, output_dim) = (4, 64)."""
        with torch.no_grad():
            out = self.model(self.x)
        self.assertEqual(out.shape, (self.batch, self.output_dim))

    def test_geen_nan_in_output(self):
        """De output bevat geen NaN-waarden (numeriek stabiel)."""
        with torch.no_grad():
            out = self.model(self.x)
        self.assertFalse(torch.isnan(out).any())

    def test_deterministisch_in_eval(self):
        """In eval-mode geeft dezelfde input twee keer dezelfde output."""
        with torch.no_grad():
            out1 = self.model(self.x)
            out2 = self.model(self.x)
        torch.testing.assert_close(out1, out2)

    def test_gradient_stroomt(self):
        """Backprop werkt: na loss.backward() krijgt minstens één parameter een gradient."""
        self.model.train()
        out = self.model(self.x)
        loss = out.sum()
        loss.backward()
        # Minstens één parameter heeft een gradient gekregen.
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad)


if __name__ == '__main__':
    unittest.main()
