import numpy as np
import unittest

from dtw import (compute_dtw_distance,
                 compute_dtw_vector)

class _BaseTest(unittest.TestCase):
    def setUp(self):
        nan = np.nan
        self.ink1 = np.array(
            [[-0.142, -1.000, 0.000, 0.000, 0.000] ,
             [-0.142, -0.899, 0.000, 1.000, 0.000] ,
             [-0.142, -0.734, 0.000, 1.000, 0.000] ,
             [-0.142, -0.443, 0.000, 1.000, 0.000] ,
             [-0.142, -0.013, 0.000, 1.000, 0.000] ,
             [-0.192, 0.354, -0.137, 0.991, 0.000] ,
             [-0.268, 0.633, -0.263, 0.965, 0.000] ,
             [-0.331, 0.823, -0.316, 0.949, 0.000] ,
             [-0.420, 0.987, -0.474, 0.880, 0.000] ,
             [-0.445, 1.000, -0.894, 0.447, 0.000] ,
             [-0.458, 0.937, -0.196, -0.981, 0.000] ,
             [-0.458, 0.797, 0.000, -1.000, 0.000] ,
             [-0.407, 0.456, 0.147, -0.989, 0.000] ,
             [-0.293, 0.253, 0.490, -0.872, 0.000] ,
             [0.010, -0.089, 0.664, -0.747, 0.000] ,
             [0.200, -0.241, 0.781, -0.625, 0.000] ,
             [0.365, -0.329, 0.880, -0.474, 0.000] ,
             [0.656, -0.405, 0.968, -0.252, 0.000] ,
             [0.719, -0.405, 1.000, 0.000, 0.000] ,
             [0.757, -0.329, 0.447, 0.894, 0.000] ,
             [0.732, 0.000, -0.077, 0.997, 0.000] ,
             [0.491, 0.456, -0.467, 0.884, 0.000] ,
             [0.352, 0.620, -0.646, 0.763, 0.000] ,
             [0.086, 0.785, -0.850, 0.526, 0.000] ,
             [-0.040, 0.835, -0.928, 0.371, 0.000] ,
             [-0.129, 0.848, -0.990, 0.141, 0.000] ,
             [-0.218, 0.848, -1.000, 0.000, 0.000] ,
             [nan, nan, nan, nan, 1.000]])

        self.ink2 = np.array(
            [[-0.266, -1.000, 0.000, 0.000, 0.000] ,
             [-0.294, -0.943, -0.447, 0.894, 0.000] ,
             [-0.294, -0.844, 0.000, 1.000, 0.000] ,
             [-0.294, -0.674, 0.000, 1.000, 0.000] ,
             [-0.280, -0.475, 0.071, 0.997, 0.000] ,
             [-0.266, -0.177, 0.048, 0.999, 0.000] ,
             [-0.180, 0.362, 0.156, 0.988, 0.000] ,
             [-0.152, 0.603, 0.117, 0.993, 0.000] ,
             [-0.124, 0.745, 0.196, 0.981, 0.000] ,
             [-0.095, 0.901, 0.179, 0.984, 0.000] ,
             [-0.095, 0.943, 0.000, 1.000, 0.000] ,
             [-0.095, 0.915, 0.000, -1.000, 0.000] ,
             [-0.095, 0.745, 0.000, -1.000, 0.000] ,
             [-0.095, 0.504, 0.000, -1.000, 0.000] ,
             [-0.081, 0.305, 0.071, -0.997, 0.000] ,
             [-0.039, 0.177, 0.316, -0.949, 0.000] ,
             [0.018, 0.050, 0.406, -0.914, 0.000] ,
             [0.103, -0.035, 0.707, -0.707, 0.000] ,
             [0.160, -0.035, 1.000, 0.000, 0.000] ,
             [0.302, 0.050, 0.857, 0.514, 0.000] ,
             [0.387, 0.220, 0.447, 0.894, 0.000] ,
             [0.472, 0.447, 0.351, 0.936, 0.000] ,
             [0.486, 0.759, 0.045, 0.999, 0.000] ,
             [0.486, 0.872, 0.000, 1.000, 0.000] ,
             [0.401, 0.957, -0.707, 0.707, 0.000] ,
             [0.132, 1.000, -0.988, 0.156, 0.000] ,
             [0.018, 1.000, -1.000, 0.000, 0.000] ,
             [-0.067, 1.000, -1.000, 0.000, 0.000] ,
             [-0.152, 1.000, -1.000, 0.000, 0.000] ,
             [nan, nan, nan, nan, 1.000]])
        
        self.ink3 = np.array(
            [[-0.435, -1.000, 0.000, 0.000, 0.000] ,
             [-0.357, -0.701, 0.252, 0.968, 0.000] ,
             [-0.331, -0.364, 0.077, 0.997, 0.000] ,
             [-0.240, 0.039, 0.220, 0.975, 0.000] ,
             [-0.188, 0.338, 0.171, 0.985, 0.000] ,
             [-0.175, 0.831, 0.026, 1.000, 0.000] ,
             [-0.175, 0.935, 0.000, 1.000, 0.000] ,
             [-0.240, 1.000, -0.707, 0.707, 0.000] ,
             [-0.292, 0.779, -0.229, -0.973, 0.000] ,
             [-0.214, 0.026, 0.103, -0.995, 0.000] ,
             [-0.058, -0.156, 0.651, -0.759, 0.000] ,
             [0.071, -0.221, 0.894, -0.447, 0.000] ,
             [0.318, -0.247, 0.995, -0.105, 0.000] ,
             [0.422, -0.182, 0.848, 0.530, 0.000] ,
             [0.487, 0.221, 0.159, 0.987, 0.000] ,
             [0.487, 0.403, 0.000, 1.000, 0.000] ,
             [0.487, 0.532, 0.000, 1.000, 0.000] ,
             [0.305, 0.766, -0.614, 0.789, 0.000] ,
             [0.162, 0.818, -0.940, 0.342, 0.000] ,
             [-0.032, 0.818, -1.000, 0.000, 0.000] ,
             [nan, nan, nan, nan, 1.000]])


class TestComputeDTWDistance(_BaseTest):
    def test_simple(self):
        self.assertAlmostEqual(
            compute_dtw_distance(self.ink1,
                                 self.ink1,
                                 alpha=0.5,
                                 penup_z=10), 
            0.000, delta=1e-3)
        self.assertAlmostEqual(
            compute_dtw_distance(self.ink1,
                                 self.ink2,
                                 alpha=0.5,
                                 penup_z=10),
            0.100, delta=1e-3)
        self.assertAlmostEqual(
            compute_dtw_distance(self.ink1,
                                 self.ink3,
                                 alpha=0.5,
                                 penup_z=10), 
            0.053, delta=1e-3)

class TestComputeDTWVector(_BaseTest):
    def test_sanity(self):
        v1 =  compute_dtw_vector(self.ink1, 
                                 self.ink3, 
                                 alpha=0.5,
                                 penup_z=10)
        self.assertEqual(v1.shape, (140,))
        
        v2 =  compute_dtw_vector(self.ink2, 
                                 self.ink3, 
                                 alpha=0.5,
                                 penup_z=10)
        self.assertEqual(v2.shape, (150,))
        
if __name__ == "__main__":
    unittest.main()