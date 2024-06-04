from acquisitions import uncnormprop_plusplus, random
import unittest
import numpy as np

class TestAcquisitions(unittest.TestCase):
    
    # selects candidate_ind.size random points
    # and outputs a probability distribution over them
    def test_uncnormprop_plusplus(self):
        acq = uncnormprop_plusplus()
        acq.set_K(10)
        u = np.random.rand(100, 2)
        candidate_ind = np.random.randint(0, 100, 10)
        acq_vals = acq.compute(u, candidate_ind)
        self.assertEqual(acq_vals.size, 10)
        self.assertTrue(np.sum(acq_vals) == 1)
