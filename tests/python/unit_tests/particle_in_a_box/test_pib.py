import qc101
import unittest


class TestParticleInABox(unittest.TestCase):

    def test_proof_of_concept(self):
        weights = [1.0]
        grid = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
        values = qc101.proof_of_concept(weights, grid, 0.1)
