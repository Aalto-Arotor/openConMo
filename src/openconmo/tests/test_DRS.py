import unittest

import sys
sys.path.append("../../")

import numpy as np

from openconmo.benchmark_methods import DRS

class TestDRS(unittest.TestCase):
    def test_drs_basic(self):
        def rmse(a, b):
            return np.sqrt(np.mean((a - b) ** 2))
        np.random.seed(0)
        N = 8192
        Delta = 500
        fs = 12e3
        t = np.linspace(0, 10, int(10*fs))
        freq = 400
        deterministic = np.sin(2 * np.pi * freq * t)
        random = np.random.normal(0, 0.30, size=t.shape)
        signal = deterministic + random

        drs_random, drs_deterministic = DRS(signal, N, Delta)
        
        rmse_random = rmse(drs_random[N+Delta:], random[N+Delta:])
        
        print(f"RMSE Random Part: {rmse_random:.6f}")
        # Check output shapes
        

        # Check that RMSE values are within reasonable bounds
        self.assertLess(rmse_random, 0.35)

        # Check that deterministic part is not all zeros
        self.assertGreater(np.linalg.norm(drs_deterministic), 0)

        # Check that random part is not all zeros
        self.assertGreater(np.linalg.norm(drs_random), 0)

if __name__ == "__main__":
    unittest.main()
