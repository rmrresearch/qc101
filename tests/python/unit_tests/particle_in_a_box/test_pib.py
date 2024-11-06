import qc101
import unittest
from math import sqrt
import numpy as np
import pickle


class TestParticleInABox(unittest.TestCase):

    def test_proof_of_concept(self):
        weights = [1.0 / sqrt(2.0), 1.0 / sqrt(2.0)]
        grid = np.linspace(0, 1.0, 100)
        values, errors = qc101.proof_of_concept(weights, grid, 0.1, 0.01, 100)

        values_file = open('values', 'wb')
        errors_file = open('errors', 'wb')
        pickle.dump(values, values_file)
        pickle.dump(errors, errors_file)

        values_file.close()
        errors_file.close()
